# Eco Video Upscaler (`ecoup`)

Công cụ AI Super-Resolution cho Video tiết kiệm VRAM và giữ nhiệt độ GPU mát mẻ, tối ưu cho dòng máy tính xách tay và GPU tầm trung (như NVIDIA GeForce RTX 3050 Laptop / 4GB VRAM).

---

## 1. Điểm nổi bật
- **Tiết kiệm VRAM (Tiling & Overlap Blending)**: Hỗ trợ video 1080p, 4K trên GPU 4GB VRAM mà không lo bị lỗi tràn bộ nhớ (Out-Of-Memory) nhờ chia nhỏ khung hình thông minh và hòa trộn viền không lộ đường nối (PSNR > 40 dB).
- **Thermal Governor (Kiểm soát nhiệt tự động)**: Tự động điều tiết tốc độ (dynamic micro-sleeps) khi GPU chạm ngưỡng nhiệt mong muốn (mặc định 70°C), tự động tạm dừng an toàn nếu quá nhiệt và khôi phục khi máy nguội.
- **Frame Deduplication (Bỏ qua khung trùng)**: Nhận diện khung hình tĩnh trong anime/phim để tái sử dụng kết quả, giảm từ 20% - 40% số lần chạy AI.
- **Checkpoint & Resume theo phân đoạn**: Chia video thành các đoạn nhỏ (mặc định 300 khung), hỗ trợ ngắt quãng và tiếp tục mà không làm mất tiến độ hay lệch tiếng.
- **NVENC Hardware Acceleration**: Tận dụng nhân mã hóa phần cứng rời của card đồ họa NVIDIA để nén video tốc độ cao (> 200 fps) mà không tốn tài nguyên tính toán CUDA.
- **Giao diện Desktop mượt mà (PyQt6)**: Giao diện tối giản chuẩn công cụ chuyên dụng (không màu mè), tách biệt luồng xử lý (non-blocking) giúp thao tác luôn mượt mà.

---

## 2. Cài đặt

### Yêu cầu hệ thống
- Hệ điều hành: Windows 10/11 hoặc Linux.
- GPU: NVIDIA GPU có hỗ trợ CUDA & NVENC (khuyên dùng RTX 3050 trở lên).
- Python: 3.10+.
- FFmpeg: Đã được cài đặt và có trong PATH (`winget install Gyan.FFmpeg`).

### Các bước cài đặt
```powershell
# Kích hoạt virtualenv
.\.venv\Scripts\Activate.ps1

# Cài đặt gói ecoup ở chế độ editable
pip install -e .
```

---

## 3. Kiểm tra môi trường (Doctor)

Chạy lệnh chẩn đoán tự động để kiểm tra FFmpeg, NVENC, cảm biến nhiệt GPU và các Execution Provider của ONNX:

```powershell
python -m ecoup.cli doctor
```

---

## 4. Hướng dẫn sử dụng

### Sử dụng qua dòng lệnh (CLI)

```powershell
# Upscale video cơ bản (mặc định scale 2x)
python -m ecoup.cli run -i input.mp4 -o output.mp4

# Upscale với độ phân giải tùy chỉnh (ví dụ 4x)
python -m ecoup.cli run -i input.mp4 -o output.mp4 --scale 4

# Tiếp tục một tác vụ bị gián đoạn trước đó
python -m ecoup.cli run -i input.mp4 -o output.mp4 --resume
```

### Khởi chạy giao diện đồ họa (GUI)

```powershell
python -m gui.main_window
```

---

## 5. Thử nghiệm Benchmark

Chạy kịch bản so sánh tốc độ, mức nhiệt và công suất tiêu thụ giữa các cấu hình:

```powershell
python scripts/benchmark.py
```

---

## 6. Chạy bộ kiểm thử (Tests)

Toàn bộ 19 test cases bao phủ video I/O, tiling, blending, dedup, thermal governor và pipeline:

```powershell
pytest
```
