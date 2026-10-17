# KẾ HOẠCH: Nâng chất lượng làm nét video (720p → 1440p, người thật)

> Tài liệu dành cho agent (Antigravity) thực thi. Làm tuần tự theo Phase. Hoàn thành Tiêu chí nghiệm thu của Phase trước khi sang Phase sau. Sau mỗi Phase: chạy `pytest`, rồi commit.
> **Phạm vi:** chỉ làm nét/upscale. **Không** làm xóa watermark trong kế hoạch này.

---

## 1. Bối cảnh

Dự án `ecoup` (thư mục `src/ecoup/`) hiện có 2 hướng upscale, cả hai đều **xử lý từng khung hình riêng lẻ** (`UpscaleEngine.upscale(frame)` trong `engine/base.py`):
- `onnx_engine.py`: model nhỏ (2xNomosUni, realesr-general-x4v3...). Kết quả: chỉ phóng pixel, không làm nét.
- `ccsr_engine.py`: CCSR diffusion chạy trên Colab (`colab_app.py`, `notebooks/EcoVideo_CCSR_Colab.ipynb`). Kết quả: vẫn mờ, người dùng không hài lòng.

**Nguyên nhân giả định:** (1) xử lý từng khung bỏ phí thông tin từ khung lân cận; (2) CCSR ưu tiên trung thực nên tạo ít chi tiết; (3) video nguồn bị nén làm mất chi tiết thật.

## 2. Mục tiêu và đầu vào/đầu ra

- **Đầu vào:** video người thật, 720p.
- **Đầu ra:** gấp đôi kích thước (1280×720 → 2560×1440), nét hơn rõ rệt so với CCSR hiện tại, **không nhấp nháy** theo thời gian, da/mắt/tóc tự nhiên (không "sáp", không khuôn mặt bị thay đổi).
- **Môi trường chạy chính:** Google Colab (GPU T4 16GB hoặc tốt hơn). Máy cục bộ RTX 3050 4GB **không** chạy nổi diffusion video; giữ engine ONNX làm "chế độ nhanh".
- **Không phải mục tiêu:** xóa watermark, nội suy khung hình, giao diện mới.

## 3. Ràng buộc

- Ngôn ngữ Python 3.10+. Giữ phong cách code và cấu trúc hiện có.
- Mọi tham số (model, tile, độ dài đoạn, overlap, cường độ hậu xử lý) phải cấu hình được qua `config.yaml` / UI, không hard-code.
- Không tải model từ nguồn không rõ. Ghi nguồn (URL repo/Hugging Face), phiên bản và checksum vào `models/README.md`.
- **Không phá vỡ** các engine và test hiện có. Chạy `pytest` sau mỗi thay đổi.
- Mọi khẳng định về VRAM/tốc độ của model phải được **đo thực tế** trong Phase 1, không dựa vào quảng cáo.

---

## Phase 0: Chuẩn bị bộ test và baseline

**Việc cần làm**
1. Tạo thư mục `eval/clips/` (thêm vào `.gitignore`, không commit video).
2. Dùng clip người dùng cung cấp (hoặc hỏi họ) để cắt 3 clip ngắn 5–8 giây ở 720p:
   - A: cận mặt, ít chuyển động.
   - B: nửa người, chuyển động nhiều.
   - C: cảnh tối hoặc nhiều nhiễu.
3. Chạy CCSR hiện tại trên 3 clip, lưu vào `eval/results/ccsr/` làm baseline.
4. Viết `scripts/eval_compare.py`: nhận các thư mục kết quả, xuất ra
   - video so sánh cạnh nhau (crop vùng mặt phóng to),
   - bảng ghi thời gian xử lý/giây video và VRAM đỉnh.

**Nghiệm thu:** có baseline CCSR + script so sánh chạy được trên 3 clip.

---

## Phase 1: Khảo sát và thử nghiệm ứng viên (quan trọng nhất)

**Mục tiêu:** chọn model theo kết quả thực tế trên clip của người dùng, **không chốt trước**.

**Ứng viên (theo thứ tự):**
1. **SeedVR2** (ByteDance) — diffusion transformer one-step cho video. Ứng viên chính.
2. **FlashVSR** — cần tra cứu repo gốc để xác nhận tình trạng, yêu cầu VRAM và giấy phép.
3. **BasicVSR++ / RealBasicVSR** — nhẹ, dùng nhiều khung lân cận. Làm mốc nhanh.
4. Chỉ thử thêm (STAR, Upscale-A-Video) nếu 3 cái trên không đạt.

> Lưu ý: nên tra cứu thêm để xem có model video SR mới hơn xuất hiện gần đây không, rồi đưa vào danh sách nếu hợp lý.

**Việc cần làm cho MỖI ứng viên**
1. Đọc README/repo gốc; ghi lại: giấy phép, VRAM yêu cầu, có bản quantized/tiling không, định dạng đầu vào.
2. Viết notebook thử nghiệm độc lập `notebooks/eval_<tên_model>.ipynb` (chưa tích hợp vào pipeline).
3. Chạy trên 3 clip, scale 2x (nếu model gốc là 4x thì resize xuống 2x bằng Lanczos).
4. Lưu kết quả vào `eval/results/<tên_model>/` và ghi số đo (thời gian/giây video, VRAM đỉnh, có OOM không, có cần tiling không) vào `docs/eval_report.md`.

**Tiêu chí đánh giá** (ghi vào `docs/eval_report.md`, kèm nhận xét ngắn):
- Độ nét mắt, mi, tóc, viền môi.
- Da: còn lỗ chân lông hay bị "sáp"/quá mịn.
- Khuôn mặt có bị thay đổi nhận dạng không.
- Nhấp nháy giữa các khung (xem video chậm, soi vùng da và tóc).
- Thời gian xử lý, độ ổn định trên Colab T4.

**Nghiệm thu:** `docs/eval_report.md` có bảng so sánh ≥ 3 model + CCSR baseline, kèm **đề xuất chọn model** có lý do. **Dừng lại và báo người dùng chọn** trước khi sang Phase 2.

---

## Phase 2: Tinh chỉnh tiền xử lý và hậu xử lý

Thực hiện trên model đã chọn.

**Việc cần làm**
1. **Tiền xử lý (tùy chọn bật/tắt):** khử block nén nhẹ trước SR (ví dụ denoise nhẹ của FFmpeg `hqdn3d` mức thấp, hoặc model khử artifact nếu có). Mục đích: không phóng to rác nén.
2. **Hậu xử lý (tùy chọn bật/tắt):**
   - Làm nét nhẹ (FFmpeg `cas` hoặc unsharp, cường độ thấp, mặc định tắt hoặc rất nhẹ).
   - Tùy chọn thêm film grain mịn để da bớt "sáp".
   - Tùy chọn trộn lại một phần ảnh gốc đã phóng bicubic (blend 0–20%) để giữ tính nhận dạng khuôn mặt.
3. Tái dùng `color_fix.py` để tránh lệch màu.
4. Chạy A/B bật/tắt từng bước trên 3 clip, ghi kết quả vào `docs/eval_report.md`, rồi đặt **giá trị mặc định** theo kết quả.

**Nghiệm thu:** có preset mặc định cho "người thật 720p→1440p", có số liệu A/B chứng minh từng bước có ích hay không (bỏ bước không ích).

---

## Phase 3: Tích hợp vào dự án

### 3.1 Thiết kế interface (quan trọng)
`UpscaleEngine.upscale(frame)` xử lý **một khung**, không phù hợp với model video cần nhiều khung. Việc cần làm:
- Thêm lớp mới trong `engine/base.py`: `VideoUpscaleEngine` với
  - `load(...)`, `close()` giống hiện tại,
  - `upscale_clip(frames: list[np.ndarray]) -> list[np.ndarray]` (nhận danh sách khung RGB uint8 của một đoạn, trả về danh sách đã upscale).
- **Giữ nguyên** `UpscaleEngine` để không phá engine cũ.
- Thêm `engine/<tên_model>_engine.py` cài đặt `VideoUpscaleEngine`, theo mẫu `ccsr_engine.py` (cách tìm repo, nạp model, chọn device, fp16).

### 3.2 Sửa pipeline (`pipeline.py`)
- Nhận biết engine kiểu clip và xử lý theo **đoạn** (tận dụng cơ chế `checkpoint.py`).
- Cấu hình mới trong `config.yaml`: `clip_length` (số khung mỗi đoạn), `clip_overlap` (số khung gối đầu).
- **Xử lý chỗ nối đoạn:** gối đầu `clip_overlap` khung giữa hai đoạn liền kề, rồi trộn tuyến tính (cross-fade) vùng gối để không bị giật/đổi màu. Với `clip_overlap` thì bỏ khung thừa khi ghép.
- Giữ `dedup`, `thermal` hoạt động với engine cũ; với engine clip, `dedup` nên tắt mặc định (khung trùng có thể phá tính liên tục thời gian) — ghi rõ trong tài liệu.
- Bảo đảm checkpoint/resume vẫn đúng: tiến độ lưu theo đoạn, ghép lại không lệch khung, không lệch tiếng.

### 3.3 Colab
- Cập nhật `colab_app.py` và notebook: thêm lựa chọn engine, preset mới, ô bật/tắt tiền/hậu xử lý.
- Cache model trong Google Drive hoặc thư mục bền vững để khỏi tải lại mỗi phiên.
- Thêm cảnh báo/xử lý khi hết GPU hoặc Colab ngắt phiên: dựa vào checkpoint để tiếp tục.
- Thêm script `scripts/download_<tên_model>.py` theo mẫu `download_ccsr.py`.

### 3.4 Test
- `tests/test_clip_engine.py`: dùng engine giả (mock) để kiểm tra chia đoạn, gối đầu, cross-fade, số khung đầu ra = số khung đầu vào.
- Cập nhật `tests/test_pipeline_smoke.py` để bao phủ nhánh engine clip.
- Không gọi model thật trong unit test.

**Nghiệm thu:** `pytest` xanh; chạy được pipeline end-to-end trên clip A với engine mới; số khung và độ dài tiếng khớp với đầu vào; chỗ nối đoạn không thấy giật bằng mắt.

---

## Phase 4: Kiểm tra cuối và tài liệu

1. Chạy lại bộ test A/B/C với cấu hình mặc định; so sánh với baseline CCSR, cập nhật `docs/eval_report.md` và `docs/benchmark.md`.
2. Chạy thử một video dài 2–3 phút: đo tổng thời gian, VRAM đỉnh, độ ổn định chỗ nối đoạn, hành vi khi ngắt và tiếp tục (resume).
3. Cập nhật `README.md`: hướng dẫn engine mới, yêu cầu GPU, preset, hạn chế đã biết.
4. Giữ engine ONNX và CCSR làm lựa chọn, ghi rõ khi nào nên dùng cái nào.

**Nghiệm thu:** README đầy đủ; báo cáo cuối có kết luận so sánh số liệu; người dùng xác nhận kết quả.

---

## 4. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Model diffusion làm khuôn mặt thay đổi so với người thật | Giảm cường độ nếu có tham số; trộn lại 10–20% ảnh gốc (Phase 2) |
| VRAM không đủ trên T4 | Dùng tiling, bản quantized, hoặc giảm `clip_length`; ghi giới hạn vào README |
| Chạy quá chậm cho video dài | Giữ BasicVSR++/RealBasicVSR làm chế độ nhanh |
| Colab ngắt phiên | Checkpoint theo đoạn đã có; kiểm thử resume ở Phase 4 |
| Giật/đổi màu ở chỗ nối đoạn | Gối đầu + cross-fade + `color_fix` (Phase 3.2) |
| Nguồn nén quá nặng | Chấp nhận giới hạn: kỳ vọng "tốt hơn rõ rệt", không phải "như quay 1440p" |
| Giấy phép model không cho phép dùng theo ý định | Kiểm tra giấy phép ở Phase 1, ghi vào báo cáo |

## 5. Quy tắc làm việc cho agent

- Làm đúng thứ tự Phase. **Dừng và hỏi người dùng** ở cuối Phase 1 (chọn model) và nếu gặp lựa chọn có tác động lớn không rõ.
- Không tự thêm tính năng ngoài phạm vi (đặc biệt: **không** làm xóa watermark).
- Báo cáo ngắn gọn sau mỗi Phase: đã làm gì, số liệu đo được, vấn đề còn tồn tại.
- Không commit video, model weight, hoặc file lớn.
