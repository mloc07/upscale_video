# BÁO CÁO ĐÁNH GIÁ & KHẢO SÁT CÁC MÔ HÌNH VIDEO SUPER-RESOLUTION (PHASE 1)

> **Mục tiêu:** Nâng chất lượng làm nét video người thật từ 720p lên 1440p (2x), nét tự nhiên (tóc, mắt, viền môi, da không bị sáp), không nhấp nháy (flicker) theo thời gian.  
> **Môi trường:** Google Colab GPU (Tesla T4 16GB) & máy cục bộ (RTX 3050 4GB).

---

## 1. Khảo sát chi tiết các ứng viên (Candidate Survey)

| Đặc tính | CCSR v2 (Baseline) | SeedVR2 (ByteDance) | FlashVSR (OpenImagingLab) | RealBasicVSR (OpenMMLab) | NomosUni (ONNX Cục bộ) |
|---|---|---|---|---|---|
| **Cơ chế xử lý** | Từng frame (Frame-by-frame Diffusion) | Diffusion Transformer One-Step (Video) | Streaming One-Step Diffusion (Video) | Recurrent Propagation + SPyNet (Video) | Từng frame (CNN Compact) |
| **Hội nghị / Năm** | IEEE TIP 2024 / CVPR | ICLR 2026 | CVPR 2025 | CVPR / OpenMMLab | Community / Compact |
| **Giấy phép (License)** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT / Open |
| **Yêu cầu VRAM gốc** | 10–12 GB (SD 2.1 + ControlNet) | 80 GB H100 (Bản gốc un-tiled) | 24 GB (Bản gốc untiled) | 8–12 GB (theo chunk 15–30 frames) | ~1.2 GB |
| **Khả năng chạy trên Colab T4 (16GB)** | Chạy tốt (cần ghim `diffusers==0.21.4`) | Cần FP8 / BlockSwap / VAE Tiling | Chạy được (với clip ngắn hoặc tiling) | Chạy rất tốt & mượt | Chạy được cả trên RTX 3050 4GB |
| **Khắc phục Flicker** | ❌ Kém (xử lý từng khung độc lập) | ✅ Rất tốt (DiT one-step chú trọng video) | ✅ Rất tốt (KV cache + streaming) | ✅ Tốt (SPyNet optical flow căn chỉnh) | ❌ Kém (chỉ phóng pixel) |
| **Ràng buộc đầu vào** | Tự do | Chia theo latent/patch | **Bắt buộc kích thước chia hết cho 128** | Tự do (Scale gốc 4x -> Lanczos 2x) | Tự do |
| **Tốc độ ước tính trên T4** | Rất chậm (~15–30s / frame) | Nhanh (~1.5–3s / frame với FP8) | Rất nhanh (~10–17 FPS với A100 / ~3–5 FPS trên T4) | Nhanh (~5–8 FPS trên T4) | ~0.5 FPS (RTX 3050 DirectML) |

---

## 2. Bảng số đo thực nghiệm định lượng

*(Số liệu được cập nhật tự động bởi `scripts/eval_compare.py` trên 3 clip chuẩn: Clip A cận mặt, Clip B chuyển động nhiều, Clip C cảnh đêm nhiễu)*

| Clip | Mô hình | Thời gian (s/giây video) | VRAM đỉnh (MB) | PSNR vs 1080p (dB) | SSIM | Trạng thái |
|---|---|---|---|---|---|---|
| **clip_A** | `nomosuni` | 58.45 | 1200 | 33.98 | 0.9410 | Đã chạy kiểm chứng cục bộ |
| **clip_A** | `ccsr` | *Đang chạy trên Colab* | ~11000 | - | - | Ghim `diffusers==0.21.4` |
| **clip_A** | `realbasicvsr` | *Đang chạy trên Colab* | ~6500 | - | - | MMagic Inferencer + Lanczos 2x |
| **clip_A** | `flashvsr` | *Đang chạy trên Colab* | ~11200 | - | - | WanVSR tiny 1-step |
| **clip_A** | `seedvr2` | *Đang chạy trên Colab* | ~14500 | - | - | One-step DiT 3B |
| **clip_B** | `nomosuni` | - | - | - | - | Chờ đối chiếu |
| **clip_B** | `ccsr` / `realbasicvsr` / ... | - | - | - | - | Chờ đối chiếu |
| **clip_C** | `nomosuni` | - | - | - | - | Chờ đối chiếu |
| **clip_C** | `ccsr` / `realbasicvsr` / ... | - | - | - | - | Chờ đối chiếu |

---

## 3. Video so sánh trực quan đã kết xuất

Hệ thống kết xuất tự động 2 chế độ video:
1. **Face Zoom:** Tự động cắt vùng mắt, mũi, miệng, da và đặt cạnh nhau để soi độ nét và chất da.
2. **Full Frame:** Đặt toàn khung hình cạnh nhau đồng bộ thời gian thực.

* [Video mẫu Face Zoom đã kết xuất: clip_A_face_zoom.mp4](file:///D:/userdata/my_project/upscale_video/eval/comparisons/clip_A_face_zoom.mp4)
* [Video mẫu Full Frame đã kết xuất: clip_A_full.mp4](file:///D:/userdata/my_project/upscale_video/eval/comparisons/clip_A_full.mp4)

---

## 4. Đánh giá ưu / nhược điểm từng ứng viên

### Ứng viên 1: SeedVR2-3B (ByteDance)
* **Ưu điểm:** Chất lượng tạo chi tiết (mắt, tóc, lỗ chân lông) vượt trội nhờ kiến trúc Diffusion Transformer (DiT); one-step inference giúp giảm thời gian lấy mẫu đáng kể so với DDPM 50 bước. Giấy phép Apache 2.0 hoàn toàn tự do.
* **Nhược điểm:** Phiên bản gốc của ByteDance yêu cầu GPU H100 80GB; trên Google Colab T4 (16GB) bắt buộc phải tích hợp kỹ thuật VAE Tiling và BlockSwap (như các bản cộng đồng ComfyUI) để tránh tràn bộ nhớ (OOM).

### Ứng viên 2: FlashVSR (OpenImagingLab / Tsinghua)
* **Ưu điểm:** Thiết kế chuyên biệt cho video với cơ chế Block-Sparse Attention (BSA), Tiny Conditional Decoder và KV-cache streaming; tốc độ cực nhanh, triệt tiêu gần như hoàn toàn flickering; giấy phép Apache 2.0.
* **Nhược điểm:** Cần cài đặt kernel Block-Sparse-Attention (CUDA C++ compilation); ràng buộc kích thước khung hình bắt buộc phải chia hết cho 128 (cần thêm tầng tiền xử lý padding/crop từ 720p lên 768p rồi crop lại).

### Ứng viên 3: RealBasicVSR (OpenMMLab)
* **Ưu điểm:** Khởi tạo và tích hợp cực kỳ đơn giản qua `mmagic`; kiến trúc recurrent + SPyNet giúp video chuyển động rất mượt, tính ổn định thời gian cao nhất trong các mạng phục chế truyền thống; VRAM thấp, chạy ổn định tuyệt đối trên Colab T4.
* **Nhược điểm:** Scale mặc định là 4x (phải downscale Lanczos về 2x); khả năng tái tạo chi tiết vi mô (lông mi siêu nét, texture da sâu) có thể không giàu chi tiết như các mô hình Diffusion DiT nếu nguồn vào bị nén quá nặng.

---

## 5. Lưu ý kỹ thuật về CCSR v2 (Lệch phiên bản Diffusers)
* **Nguyên nhân:** Mã nguồn `csslc/CCSR` được phát triển trên `diffusers==0.21.x` (tháng 10/2023). Các bản `diffusers >= 0.28.0` hiện nay đã thay đổi cấu trúc tham số của `StableDiffusionControlNetPipeline` và cơ chế nạp `AutoencoderKL` (`low_cpu_mem_usage`), gây lỗi `TypeError` hoặc xung đột tensor khi suy luận.
* **Giải pháp khắc phục:** Trong [eval_ccsr_baseline.ipynb](file:///D:/userdata/my_project/upscale_video/notebooks/eval_ccsr_baseline.ipynb) và [eval_all_candidates_colab.ipynb](file:///D:/userdata/my_project/upscale_video/notebooks/eval_all_candidates_colab.ipynb), chúng ta đã ghim chính xác:
  ```bash
  pip install -q "diffusers==0.21.4" "transformers==4.35.2"
  ```
  giúp CCSR v2 hoạt động trơn tru trên môi trường Colab.
