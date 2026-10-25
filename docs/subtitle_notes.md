# Ghi chú Kỹ thuật & Xác minh Môi trường: Module Phụ đề Tự động (Subtitle)

Tài liệu ghi lại kết quả xác minh kỹ thuật thực tế phục vụ module Phụ đề tự động (`ecoup.subtitle`):
- **Ngày kiểm tra:** 25/10/2026
- **Môi trường xác minh:** Google Colab (Tesla T4 16GB, Ubuntu 22.04 LTS, Python 3.13 / Python 3.10) & Windows 11 (CUDA 12).

---

## 1. Xác minh FFmpeg & libass (Ghép phụ đề cứng)

### Kiểm tra Bộ lọc FFmpeg
Lệnh kiểm tra: `ffmpeg -hide_banner -filters | grep -E ' (subtitles|ass) '`
- **Kết quả:**
  - `subtitles`: Hỗ trợ render text subtitles lên video sử dụng thư viện `libass`.
  - `ass`: Hỗ trợ render Advanced SubStation Alpha subtitles đầy đủ kiểu dáng (font, size, margin, alignment, outline, shadow) sử dụng `libass`.

### Kiểm tra Phông chữ Tiếng Việt (Unicode Diacritics)
- **Trên Linux / Colab:**
  - Cài đặt: `apt-get install -y fonts-noto-core fonts-noto-cjk fontconfig`
  - Lệnh xác minh: `fc-list ":family=Noto Sans" family`
  - Kết quả: `Noto Sans` có sẵn và hiển thị đầy đủ 100% các ký tự tiếng Việt có dấu (`à, á, ả, ã, ạ, ă, â, đ, ê, ô, ơ, ư...`).
- **Trên Windows:**
  - Hệ thống tự động sử dụng font hệ thống `Arial` hoặc `Segoe UI` (đều có sẵn trong `C:\Windows\Fonts\` và hỗ trợ hoàn hảo tiếng Việt).

---

## 2. Xác minh Thư viện ASR (faster-whisper & ctranslate2)

- **Thư viện:** `faster-whisper>=1.0.0` chạy trên nền backend tối ưu hóa `ctranslate2`.
- **Mô hình khuyến nghị:** `large-v3` (đa ngôn ngữ, độ chính xác cao nhất).
- **Tương thích CUDA 12 & cuDNN 9:**
  - Trên Colab, PyTorch/CUDA môi trường có thể dùng CUDA 13 hoặc cuDNN khác, khiến `ctranslate2` thiếu `libcublas.so.12` / `libcudnn.so.9`.
  - Giải pháp đã tích hợp: Hàm `_preload_cuda12_libs()` trong `asr.py` tự động nạp các thư viện từ bánh xe pip `nvidia-cublas-cu12` và `nvidia-cudnn-cu12` thông qua `ctypes.CDLL(RTLD_GLOBAL)`.
- **Giải phóng bộ nhớ GPU sau khi nhận dạng:**
  - Mô hình Whisper được nạp, nhận dạng giọng nói, và lập tức giải phóng bằng `del model`, `gc.collect()`, `torch.cuda.empty_cache()` để nhường 100% VRAM GPU cho các tác vụ khác.
- **Lọc tạp âm và khoảng lặng (VAD):**
  - Kích hoạt `vad_filter=True` kết hợp thuật toán Silero VAD với `min_silence_duration_ms=500` giúp loại bỏ triệt để hiện tượng lặp từ ("bịa chữ") ở các đoạn im lặng hoặc nhạc nền không lời.

---

## 3. Xác minh Google GenAI SDK & Gemini Model IDs

- **SDK chính thức:** `google-genai` (chuẩn unified SDK mới nhất của Google, import qua `from google import genai`).
- **Khóa API (`GEMINI_API_KEY`):**
  - Được đọc trực tiếp từ biến môi trường `GEMINI_API_KEY` hoặc truyền qua giao diện.
  - Tuyệt đối không lưu khóa API trong mã nguồn, git repository hoặc file cấu hình.
- **Danh sách Model ID đã kiểm tra qua `client.models.list()`:**
  - `gemini-3.1-flash-lite` (Mô hình dịch nhanh, nhẹ, chi phí thấp, khuyên dùng mặc định).
  - `gemini-3.5-flash-lite` (Mô hình dự phòng khi 3.1 gặp tải cao).
  - `gemini-2.5-flash-lite` (Mô hình dự phòng thế hệ trước).
  - `gemini-3.5-flash` / `gemini-2.5-flash`.
- **Cơ chế chịu lỗi & Tự động chuyển vùng (Auto Fallback):**
  - Khi Google API phản hồi mã `503 UNAVAILABLE` (mô hình đang quá tải tạm thời), hệ thống tự động retry với exponential backoff.
  - Nếu sau số lần retry vẫn bị quá tải, hệ thống tự động chuyển tiếp qua model dự phòng trong danh sách `fallback_models` (`gemini-3.5-flash-lite` $\rightarrow$ `gemini-2.5-flash-lite`) mà không làm gián đoạn tiến trình.
- **Đầu ra có cấu trúc (Structured Outputs):**
  - Cấu hình: `response_mime_type="application/json"`, kết hợp `response_schema` kiểm soát mảng JSON `[{"id": 0, "text": "..."}]` khớp 100% số lượng câu thoại đầu vào.
- **Bộ nhớ đệm theo lô (Cache on Disk):**
  - Toàn bộ kết quả dịch được băm SHA-256 lưu trong thư mục `--cache-dir`. Khi bị ngắt kết nối mạng hoặc chạy lại, các đoạn đã dịch sẽ được đọc tức thì từ cache mà không tốn thêm request API.
