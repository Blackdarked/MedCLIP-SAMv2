# Báo cáo LaTeX — MedCLIP-SAMv2

Cấu trúc theo yêu cầu GV:
1. **Giới thiệu** — bài toán, khó khăn, ý tưởng MedCLIP-SAMv2, mục tiêu đồ án  
2. **Phương pháp thực hiện** — pipeline 3 giai đoạn, dữ liệu, chỉ số  
3. **Thực nghiệm** — kết quả bài báo gốc (arXiv:2409.19483), tái hiện, mở rộng (hình + phân tích)

Tham khảo khung trình bày: `Oldpaper.pdf` (báo cáo nhóm trước — dự báo cổ phiếu).

## Overleaf

1. **Compiler:** Menu → Settings → **XeLaTeX** (bắt buộc vì `fontspec`).
2. Upload **toàn bộ** thư mục sau:

```
main.tex
references.tex
images/
  brain_sample_0001.png
  xray_sample_0001.png
  breast_sample.png
  ct_sample.png
  breast_summary_best.png
  breast_summary_worst.png
  brain_summary_best.png
  brain_summary_worst.png
  xray_summary_best.png
  xray_summary_worst.png
  ct_summary_best.png
  ct_summary_worst.png
```

3. Recompile 2 lần nếu cần cập nhật tham chiếu hình/bảng.

## Cursor (LaTeX Workshop)

Đã cài **MiKTeX 25.12**; LaTeX Workshop trỏ tới:
`C:\Users\PC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe`

1. **Reload Cursor:** `Ctrl+Shift+P` → `Developer: Reload Window`
2. Mở `Paper/main.tex` → `Ctrl+Alt+B` (build) → `Ctrl+Alt+V` (PDF)

Lần build đầu MiKTeX có thể tải thêm gói (IEEEtran, babel-vietnamese, …)—chờ vài phút.

Nếu vẫn báo `'xelatex' is not recognized`, đóng hẳn Cursor rồi mở lại (PATH mới).

## Cập nhật hình sau khi chạy lại pipeline

```powershell
cd d:\Study\IS252\Project\MedCLIP-SAMv2
python visualize.py --all
# Rồi copy lại từ working/visualizations/*/summary_*.png vào Paper/images/
```

## Tác giả

Đã điền theo `Oldpaper.pdf` (4 SV + ThS. Dương Phi Long). Sửa trong `main.tex` nếu GV đổi nhóm.
