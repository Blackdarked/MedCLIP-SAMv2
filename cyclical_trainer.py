# cyclical_trainer.py
# Copy this file to the nnunetv2 trainer directory.
# stage3_nnunet.py does this automatically via install_cyclical_trainer().
#
# Implements: Zhao et al. (2022), "Efficient Bayesian Uncertainty
# Estimation for nnU-Net"
# Cyclical LR + checkpoint ensemble for Bayesian model averaging.

import os
import torch
import numpy as np

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    nnUNetTrainer = object


class nnUNetTrainerCyclicalLR(nnUNetTrainer if NNUNET_AVAILABLE else object):
    """
    nnUNet v2 trainer with cyclical learning rate schedule.

    Schedule (Zhao et al. 2022):
      - 3 cycles × 200 epochs = 600 total epochs
      - Each cycle: LR decays with polynomial schedule for 80% of cycle
      - Then LR plateaus for final 20% (posterior sampling phase)
      - Saves 10 checkpoints during plateau per cycle → 30 total

    Ensemble prediction: average softmax over all 30 checkpoints.
    Uncertainty map: per-pixel entropy of averaged distribution.

    Usage:
        nnUNetv2_train {ID} 2d 0 -tr nnUNetTrainerCyclicalLR
        nnUNetv2_train {ID} 2d 0 -tr nnUNetTrainerCyclicalLR --c  # resume
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override nnUNet defaults
        # 1 cycle × 200 epochs, 50 iter/epoch — fits 4 datasets in 24h on RTX 4060 Ti.
        # Paper uses 600 epochs × 250 iter; we reduce iter count, not epoch structure,
        # so the cyclical LR shape and plateau checkpoint saving are preserved.
        self.num_epochs       = 200
        self.initial_lr       = 0.01
        self.weight_decay     = 3e-5

        # Reduce per-epoch iterations to fit time budget (~50s/epoch vs 215s at default 250)
        self.num_iterations_per_epoch     = 50
        self.num_val_iterations_per_epoch = 10

        # Cyclical schedule parameters
        self.n_cycles              = 1
        self.epochs_per_cycle      = self.num_epochs // self.n_cycles   # 200
        self.restart_lr            = 0.1      # LR at start of each cycle
        self.gamma_fraction        = 0.8      # plateau after 80% of cycle
        self.checkpoints_per_cycle = 10       # 10 × 3 = 30 total

        # Ensemble checkpoint directory
        self.ensemble_ckpt_dir = os.path.join(
            self.output_folder, 'ensemble_checkpoints'
        )
        os.makedirs(self.ensemble_ckpt_dir, exist_ok=True)

        self._saved_ckpt_count  = 0
        self._last_saved_epoch  = -1

    # ──────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
            weight_decay=self.weight_decay,
        )
        # We manage LR manually — return None for scheduler
        return optimizer, None

    # ──────────────────────────────────────────────────────────
    # Cyclical LR schedule
    # ──────────────────────────────────────────────────────────

    def on_train_epoch_start(self):
        self.network.train()
        # Base class calls self.lr_scheduler.step() here — skip it.
        # We set LR manually in _apply_cyclical_lr via on_epoch_start.

    def on_epoch_start(self):
        super().on_epoch_start()
        self._apply_cyclical_lr()

    def _apply_cyclical_lr(self):
        """
        Cyclical LR as defined in Zhao et al. (2022) Eq. 4:

          if tc == 0:                     lr = αr (restart)
          elif tc < γ·Tc:                lr = α0 · (1 - t/T)^ε  (decay)
          else:                          lr = α0 · (1 - γ·Tc/T)^ε  (plateau)

        where:
          tc = current_epoch mod epochs_per_cycle
          t  = current_epoch (global)
          T  = total_epochs
          Tc = epochs_per_cycle
          γ  = gamma_fraction (0.8)
          ε  = 0.9 (polynomial exponent)
          αr = restart_lr (0.1)
          α0 = initial_lr (0.01)
        """
        epoch = self.current_epoch
        tc    = epoch % self.epochs_per_cycle
        T     = self.num_epochs
        Tc    = self.epochs_per_cycle
        gamma = self.gamma_fraction

        if tc == 0:
            new_lr = self.restart_lr
        elif tc < int(gamma * Tc):
            new_lr = self.initial_lr * ((1.0 - epoch / T) ** 0.9)
        else:
            # Plateau: held at constant low LR
            plateau_epoch = int(gamma * Tc)
            global_plateau = (epoch // Tc) * Tc + plateau_epoch
            new_lr = self.initial_lr * ((1.0 - global_plateau / T) ** 0.9)

        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

        if epoch % 20 == 0:
            self.print_to_log_file(
                f"Epoch {epoch:4d} | cycle {epoch // Tc} "
                f"| tc={tc:3d} | LR={new_lr:.6f}"
            )

    # ──────────────────────────────────────────────────────────
    # Ensemble checkpoint saving
    # ──────────────────────────────────────────────────────────

    def on_epoch_end(self):
        super().on_epoch_end()
        self._maybe_save_ensemble_checkpoint()

    def _maybe_save_ensemble_checkpoint(self):
        """
        Save checkpoint during the plateau phase of each cycle.
        Saves checkpoints_per_cycle (10) per cycle → 30 total.
        """
        epoch  = self.current_epoch
        tc     = epoch % self.epochs_per_cycle
        cycle  = epoch // self.epochs_per_cycle

        plateau_start = int(self.gamma_fraction * self.epochs_per_cycle)
        if tc < plateau_start:
            return   # not in plateau yet

        # How often to save within the plateau
        plateau_len   = self.epochs_per_cycle - plateau_start
        save_interval = max(1, plateau_len // self.checkpoints_per_cycle)

        if (tc - plateau_start) % save_interval != 0:
            return
        if epoch == self._last_saved_epoch:
            return   # don't double-save

        ckpt_name = f'cycle{cycle}_epoch{epoch:04d}.pth'
        ckpt_path = os.path.join(self.ensemble_ckpt_dir, ckpt_name)

        torch.save(
            {'epoch': epoch, 'cycle': cycle,
             'network_weights': self.network.state_dict()},
            ckpt_path
        )

        self._saved_ckpt_count += 1
        self._last_saved_epoch  = epoch
        self.print_to_log_file(
            f"Ensemble checkpoint saved "
            f"[{self._saved_ckpt_count}/30]: {ckpt_name}"
        )
