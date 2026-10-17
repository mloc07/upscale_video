# Eco Video Upscaler (`ecoup`)

Hệ thống AI Super-Resolution cho Video tiết kiệm VRAM và kiểm soát nhiệt độ GPU thông minh, tối ưu hóa cho cả máy tính xách tay cá nhân (như NVIDIA GeForce RTX 3050 Laptop / 4GB VRAM) lẫn môi trường điện toán đám mây Google Colab (Tesla T4 16GB).

---

## 1. Điểm nổi bật & Tính năng mới

- **Kiến trúc Video Super-Resolution (SeedVR2-3B DiT)**: Tích hợp mô hình Diffusion Transformer chuyên phục chế video của ByteDance (ICLR 2026), khử triệt để hiện tượng nhấp nháy ánh sáng (temporal flicker), tái tạo chi tiết da người chân thực (sợi tóc, bờ mi, khóe môi, vân da) mà không bị "bệt/nhựa".
- **Hệ thống Clip Chunking & Cross-Fade Blending**: Xử lý video theo từng cụm thời gian (`clip_length: 16`, `clip_overlap: 2`) kết hợp hòa trộn tuyến tính (cross-fade) tại vùng biên gối đầu, đảm bảo chuyển động liền mạch và bảo toàn 100% số lượng khung hình gốc.
- **Bộ Preset Tối ưu Chân dung & Nhận dạng**:
  - *Denoise (0.10)*: Khử viền răng cưa và artifact nén 720p trước khi đưa vào mạng AI.
  - *Wavelet Color Fix*: Triệt tiêu sai lệch màu sắc (Delta-E giảm từ 3.91 xuống 3.37) và ổn định quang độ.
  - *Identity Detail Recovery (0.10)*: Hòa trộn 10% ảnh gốc nhằm bảo toàn tuyệt đối giải phẫu khuôn mặt và nụ cười.
  - *Unsharp Masking (0.15)*: Tách bạch chân thực từng sợi tóc và lông mi.
- **Tiết kiệm VRAM (Tiling & Overlap Blending)**: Hỗ trợ xử lý video 1080p, 1440p, 4K trên GPU 4GB VRAM mà không lo bị tràn bộ nhớ (Out-Of-Memory) nhờ chia mảnh thông minh (PSNR > 40 dB).
- **Thermal Governor (Kiểm soát nhiệt tự động)**: Tự động điều tiết tốc độ (dynamic micro-sleeps) khi GPU chạm ngưỡng nhiệt mong muốn (mặc định 70°C), tự động tạm dừng an toàn nếu quá nhiệt và khôi phục khi máy nguội.
- **Checkpoint & Resume theo phân đoạn**: Tự động lưu tiến độ định kỳ (mặc định mỗi 300 khung) vào file `state.json`. Khi bị ngắt quãng (hoặc Colab timeout), hệ thống tiếp tục xử lý chính xác từ khung gián đoạn mà không làm mất thời gian hay lệch tiếng.
- **NVENC Hardware Acceleration**: Tận dụng nhân mã hóa phần cứng của card đồ họa NVIDIA để xuất video thành phẩm tốc độ cao.

---

## 2. Hướng dẫn chọn Mô hình AI (Model Selection)

| Mô hình | Môi trường tối ưu | Ưu điểm nổi bật | Khi nào nên dùng? |
|---|---|---|---|
| 👑 **SeedVR2-3B (DiT)** | Google Colab (Tesla T4 16GB / A100) | Triệt tiêu flicker, da người cực nét tự nhiên, giữ trọn nhận dạng | **Khuyên dùng số 1** cho video người thật, chân dung, vlog, podcast từ 720p lên 1440p |
| ⚡ **NomosUni 2x / RealESRGAN** | Cục bộ (RTX 3050 Laptop 4GB) | VRAM chỉ ~1.2 – 1.8 GB, GPU cực mát (<70°C), tốc độ cao | Phù hợp xem trước (local preview) trên máy cá nhân hoặc video hoạt hình/anime |
| ⚙️ **CCSR v2 (Diffusion)** | Google Colab (Tesla T4 16GB) | Kiểm soát cấu trúc qua ControlNet (ghim `diffusers==0.21.4`) | Dự phòng khi muốn can thiệp sâu vào cấu trúc hình ảnh qua prompt/control strength |

---

## 3. Cài đặt hệ thống

### Yêu cầu tiên quyết
- Hệ điều hành: Windows 10/11 hoặc Linux (Ubuntu 22.04+).
- GPU: NVIDIA GPU hỗ trợ CUDA (RTX 3050 trở lên cho máy cục bộ; Tesla T4 trở lên cho Colab).
- Python: 3.10 – 3.11.
- FFmpeg: Đã được cài đặt và có trong PATH (`winget install Gyan.FFmpeg` trên Windows).

### Cài đặt môi trường
```powershell
# Kích hoạt virtualenv
.\.venv\Scripts\Activate.ps1

# Cài đặt gói ecoup ở chế độ editable
pip install -e .
```

### Kiểm tra chẩn đoán hệ thống
```powershell
python -m ecoup.cli doctor
```

---

## 4. Hướng dẫn sử dụng

### 4.1. Giao diện WebUI trên Google Colab / Cloud
Chạy ứng dụng Gradio tương thích hoàn toàn với Google Colab, hỗ trợ tạo liên kết công khai `gradio.live`:
```bash
python colab_app.py --share
```
*Giao diện cung cấp đầy đủ các thanh trượt điều chỉnh: Denoise, Detail Recovery, Sharpen, Film Grain, lựa chọn mô hình SeedVR2/CCSR/NomosUni, và nút Hủy an toàn.*

### 4.2. Giao diện Desktop (PyQt6)
Dành cho máy tính cá nhân cục bộ:
```powershell
python -m gui.main_window
```

### 4.3. Dòng lệnh (CLI)
```powershell
# Upscale video cơ bản (mặc định 2x)
python -m ecoup.cli run -i input.mp4 -o output.mp4

# Upscale với tỷ lệ tùy chỉnh (ví dụ 4x)
python -m ecoup.cli run -i input.mp4 -o output.mp4 --scale 4

# Tiếp tục tác vụ bị gián đoạn từ checkpoint
python -m ecoup.cli run -i input.mp4 -o output.mp4 --resume
```

---

## 5. Tải trọng số mô hình (Model Weights)

```powershell
# Tải bộ trọng số SeedVR2-3B
python scripts/download_seedvr2.py

# Tải bộ trọng số CCSR v2 (ControlNet)
python scripts/download_ccsr.py
```

---

## 6. Đánh giá & Kiểm thử (Evaluation & Tests)

### Chạy toàn bộ bộ kiểm thử tự động
Toàn bộ 32 test cases kiểm tra video I/O, tiling, blending, dedup, checkpoint, clip chunking, và pipeline:
```powershell
pytest -v
```

### Đánh giá so sánh đa mô hình (Side-by-Side Face Zoom & Full Frame)
```powershell
python scripts/eval_compare.py --models nomosuni seedvr2
```

### Thử nghiệm video dài (Long Video Stress Test 2.5 phút)
```powershell
python scripts/benchmark_long_video.py
```

---

## 7. Hạn chế đã biết & Lưu ý kỹ thuật

1. **Phiên bản thư viện CCSR**: Mã nguồn CCSR v2 yêu cầu chính xác `diffusers==0.21.4` và `transformers==4.35.2`. Phiên bản mới hơn sẽ gây lỗi chữ ký tham số trong `StableDiffusionControlNetPipeline`.
2. **Quản lý bộ nhớ VRAM cho SeedVR2**: Bản un-tiled gốc của ByteDance nhắm tới GPU H100 80GB. Trên Tesla T4 16GB, bắt buộc dùng trọng số FP8, VAE Tiling, và hoán đổi khối (block swap) để tránh tràn bộ nhớ.
3. **Deduplication trên Video Engine**: Mặc định tính năng Dedup sẽ tự động tắt khi sử dụng các engine xử lý theo clip (`is_clip_engine`) để bảo vệ tính liên tục thời gian của mô hình diffusion.
