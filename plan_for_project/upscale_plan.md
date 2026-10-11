# KẾ HOẠCH DỰ ÁN: Eco Video Upscaler (AI upscale video tiết kiệm GPU)

> Tài liệu này dành cho agent (Antigravity) thực thi. Hãy làm tuần tự theo từng Phase, hoàn thành tiêu chí nghiệm thu (Acceptance Criteria) của Phase trước khi sang Phase sau. Sau mỗi Phase, chạy test và commit.

---

## 1. Mục tiêu

Xây dựng công cụ upscale video bằng AI (ví dụ 720p → 1080p, 1080p → 4K) với ưu tiên hàng đầu:

1. **Tiết kiệm phần cứng**: dùng ít VRAM, ít GPU utilization, ít điện.
2. **Giữ nhiệt thấp**: tự động giảm tải khi GPU nóng.
3. **Chất lượng chấp nhận được**: tốt hơn rõ rệt so với bicubic/Lanczos.
4. **Có thể tạm dừng / tiếp tục** (resume) để GPU nghỉ và không mất tiến độ.

**Không phải mục tiêu**: chất lượng tối đa tuyệt đối, diffusion-based upscaling, nội suy khung hình (frame interpolation).

## 2. Ràng buộc & giả định

- Hệ điều hành mục tiêu: Windows 10/11 (ưu tiên), Linux (hỗ trợ nếu dễ).
- GPU: NVIDIA tầm trung (ví dụ RTX 3050–4060, 4–8 GB VRAM). Phần mềm phải chạy được cả khi chỉ có CPU (chậm) hoặc GPU không phải NVIDIA (fallback).
- Ngôn ngữ: **Python 3.10+**.
- GUI (Phase 6): **PyQt6** hoặc **PySide6** (chọn một, dùng nhất quán).
- Mọi giá trị ngưỡng (nhiệt độ, công suất, tile size...) phải **cấu hình được** qua file `config.yaml`, không hard-code.
- Không tải model từ nguồn không rõ; ghi rõ URL/nguồn và checksum của model trong `models/README.md`.

## 3. Công nghệ sử dụng

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Decode/Encode video | **FFmpeg** (subprocess + pipe) | Encode bằng **NVENC** (`h264_nvenc` / `hevc_nvenc`) nếu có, fallback `libx264` |
| Inference | **ONNX Runtime** (CUDA EP, tùy chọn TensorRT EP) | Fallback PyTorch FP16 nếu cần |
| Nạp model | **Spandrel** (đọc nhiều kiến trúc) hoặc ONNX đã export sẵn | Dùng để export sang ONNX |
| Model mặc định (nhẹ) | `realesr-animevideov3` (anime), `realesr-general-x4v3` (tổng quát), Compact/SPAN | Tránh RRDB 23-block, SwinIR, HAT, diffusion |
| Giám sát GPU | **pynvml** (nvidia-ml-py) | Đọc nhiệt độ, công suất, utilization, VRAM |
| Phát hiện cảnh | **PySceneDetect** | Tùy chọn, cho tối ưu khung |
| Xử lý ảnh | NumPy, OpenCV (opencv-python-headless) | |
| Cấu hình | PyYAML | |
| CLI | `argparse` hoặc `typer` | |
| Test | `pytest` | |
| Logging | `logging` chuẩn + `rich` (tùy chọn) | |

## 4. Cấu trúc thư mục đề xuất

```
eco-upscaler/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.yaml
├── models/
│   └── README.md            # nguồn, license, checksum model
├── src/
│   └── ecoup/
│       ├── __init__.py
│       ├── cli.py            # điểm vào dòng lệnh
│       ├── config.py         # nạp & validate config
│       ├── pipeline.py       # điều phối decode → xử lý → encode
│       ├── video_io.py       # FFmpeg reader/writer qua pipe
│       ├── engine/
│       │   ├── base.py       # interface UpscaleEngine
│       │   ├── onnx_engine.py
│       │   └── torch_engine.py   # fallback
│       ├── tiling.py         # chia tile + ghép (overlap, blend)
│       ├── dedup.py          # phát hiện khung trùng/tĩnh
│       ├── thermal.py        # ThermalGovernor (pynvml)
│       ├── checkpoint.py     # lưu/khôi phục tiến độ
│       └── utils.py
├── gui/                      # Phase 6
│   └── main_window.py
├── tests/
│   ├── test_tiling.py
│   ├── test_dedup.py
│   ├── test_thermal.py
│   ├── test_checkpoint.py
│   └── test_pipeline_smoke.py
└── scripts/
    ├── export_onnx.py        # chuyển model .pth → .onnx
    └── benchmark.py
```

## 5. Cấu hình mẫu (`config.yaml`)

```yaml
input: null                 # đường dẫn video, có thể truyền qua CLI
output: null

model:
  name: realesr-general-x4v3
  path: models/realesr-general-x4v3.onnx
  scale: 4                  # scale gốc của model
  target_scale: 2           # scale đầu ra mong muốn (resize phần còn lại)
  precision: fp16           # fp16 | fp32

inference:
  provider: auto            # auto | cuda | tensorrt | cpu
  tile_size: 256            # 0 = không chia tile
  tile_overlap: 16
  batch_size: 1

frame_saving:
  enable_dedup: true
  dedup_threshold: 0.5      # MSE nhỏ hơn ngưỡng này thì coi là khung trùng
  enable_scene_detect: false

thermal:
  enabled: true
  poll_interval_sec: 2
  temp_soft_c: 70           # vượt ngưỡng này: bắt đầu throttle
  temp_hard_c: 80           # vượt ngưỡng này: tạm dừng đến khi nguội
  temp_resume_c: 65         # nguội xuống ngưỡng này thì chạy lại
  max_gpu_util_pct: 85
  throttle_sleep_ms_min: 0
  throttle_sleep_ms_max: 500
  power_limit_watts: null   # null = không đụng; cần quyền admin nếu đặt

encode:
  codec: auto               # auto | h264_nvenc | hevc_nvenc | libx264
  crf_or_cq: 21
  preset: p4                # preset NVENC
  copy_audio: true

checkpoint:
  enabled: true
  dir: .ecoup_state
  interval_frames: 300
```

## 6. Kế hoạch theo Phase

### Phase 0 — Khởi tạo dự án
**Công việc**
- Tạo cấu trúc thư mục, `pyproject.toml`, `requirements.txt`, `.gitignore` (bỏ qua `models/*.onnx`, `.ecoup_state/`, `*.mp4` mẫu).
- Thiết lập virtualenv, cài phụ thuộc.
- Viết `config.py` nạp YAML, validate kiểu dữ liệu và giá trị hợp lệ, báo lỗi rõ ràng.
- Hàm kiểm tra môi trường: có FFmpeg trong PATH không, có NVENC không (`ffmpeg -encoders`), có NVIDIA GPU/pynvml không, ONNX Runtime có provider nào.

**Nghiệm thu**
- `python -m ecoup.cli doctor` in ra báo cáo môi trường (FFmpeg, NVENC, GPU, providers) và không crash khi thiếu GPU.

### Phase 1 — Video I/O qua FFmpeg
**Công việc**
- `video_io.py`: dùng `ffprobe` đọc metadata (độ phân giải, fps, số khung, codec, audio).
- Reader: đọc khung thô `rgb24` qua `stdout` của FFmpeg thành mảng NumPy `(H, W, 3)`.
- Writer: ghi khung thô vào `stdin` của FFmpeg, encode bằng NVENC (fallback x264), **mux audio gốc** từ file nguồn.
- Xử lý đúng: fps lẻ (29.97), pixel aspect, rotation metadata, đóng pipe an toàn khi lỗi/huỷ.

**Nghiệm thu**
- Chạy "passthrough" (đọc → ghi không xử lý) cho video mẫu 30 giây, kết quả cùng độ dài, có âm thanh, đồng bộ.

### Phase 2 — Engine suy luận (inference)
**Công việc**
- `engine/base.py`: interface `UpscaleEngine` với `load()`, `upscale(frame: np.ndarray) -> np.ndarray`, `close()`.
- `engine/onnx_engine.py`: nạp ONNX bằng ONNX Runtime, chọn provider (TensorRT → CUDA → CPU), hỗ trợ FP16 input/output, dùng IO binding nếu có lợi.
- `scripts/export_onnx.py`: nạp model `.pth` bằng Spandrel, export ONNX với trục động (dynamic axes) cho H, W.
- `tiling.py`: chia khung thành tile (kích thước & overlap cấu hình), upscale từng tile, ghép lại với **blend ở vùng overlap** để không lộ đường nối. Xử lý biên (padding reflect).
- Nếu `target_scale` < `scale` của model: resize kết quả về đúng kích thước đích bằng Lanczos (`cv2.INTER_LANCZOS4`).
- `engine/torch_engine.py`: fallback PyTorch `half()` (làm sau cùng, tuỳ chọn).

**Nghiệm thu**
- Upscale 1 ảnh mẫu: không có đường nối tile nhìn thấy được; tile_size=0 và tile_size=256 cho kết quả gần giống nhau (PSNR giữa hai kết quả > 40 dB).
- Test `test_tiling.py` pass (kích thước đúng, không lệch pixel, blend overlap đúng).
- Đo và log: VRAM đỉnh, thời gian/khung.

### Phase 3 — Tiết kiệm khung hình (giảm số lần chạy model)
**Công việc**
- `dedup.py`: so sánh khung hiện tại với khung trước (thu nhỏ về ~160px xám, tính MSE hoặc hash cảm quan). Nếu dưới ngưỡng → **tái sử dụng kết quả upscale của khung trước**, không chạy model.
- Thống kê: số khung bỏ qua / tổng, in vào log cuối job.
- (Tùy chọn) `PySceneDetect`: reset bộ so sánh khi sang cảnh mới để tránh dùng lại sai khung.
- Cẩn trọng: chỉ dedup khi khung thật sự gần như giống nhau; ngưỡng bảo thủ mặc định để tránh hiện tượng "đứng hình" giả.

**Nghiệm thu**
- Trên video có cảnh tĩnh (slide, màn hình), tỷ lệ khung bỏ qua > 0 và video đầu ra không có hiện tượng giật/đứng bất thường khi xem thử.
- `test_dedup.py` pass (khung giống nhau → trùng; khung khác → không trùng).

### Phase 4 — Kiểm soát nhiệt & công suất (`ThermalGovernor`)
**Công việc**
- `thermal.py`: chạy thread nền đọc `pynvml` mỗi `poll_interval_sec`: nhiệt độ, GPU utilization, công suất, VRAM.
- Cơ chế throttle theo mức:
  - Dưới `temp_soft_c`: không sleep.
  - Giữa `temp_soft_c` và `temp_hard_c`: sleep giữa các khung tăng tuyến tính từ `throttle_sleep_ms_min` đến `throttle_sleep_ms_max`.
  - Trên `temp_hard_c`: **tạm dừng hẳn** đến khi nhiệt độ ≤ `temp_resume_c`, rồi tiếp tục.
  - Nếu utilization > `max_gpu_util_pct`: thêm sleep nhẹ.
- Pipeline gọi `governor.wait_if_needed()` trước mỗi khung (hoặc mỗi batch).
- Nếu không có NVIDIA GPU / pynvml lỗi: governor tự vô hiệu hoá, ghi cảnh báo, **không crash**.
- Tùy chọn `power_limit_watts`: gọi `nvidia-smi -pl <W>` khi có quyền admin; nếu thất bại chỉ cảnh báo, hướng dẫn người dùng tự đặt thủ công. Khôi phục giá trị cũ khi kết thúc nếu đã đổi.
- Ghi log nhiệt độ/công suất theo thời gian vào CSV (`.ecoup_state/telemetry.csv`) để vẽ đồ thị sau.

**Nghiệm thu**
- `test_thermal.py` dùng mock pynvml: mô phỏng nhiệt tăng → governor sleep/pause đúng ngưỡng và resume đúng.
- Chạy thật 5 phút: nhiệt độ GPU không vượt `temp_hard_c` đáng kể; telemetry CSV được ghi.

### Phase 5 — Pipeline hoàn chỉnh, CLI, checkpoint
**Công việc**
- `pipeline.py`: nối Reader → (dedup) → governor → engine → Writer; hỗ trợ huỷ (Ctrl+C) sạch sẽ, dọn tiến trình FFmpeg.
- Dùng hàng đợi (producer/consumer) giữa decode, inference, encode để chồng lấn I/O và tính toán nhưng giới hạn kích thước hàng đợi nhỏ (tránh tốn RAM).
- `checkpoint.py`: định kỳ lưu số khung đã xử lý và cách ghép; hỗ trợ `--resume`: xử lý theo **đoạn (segment)** (ví dụ mỗi 300 khung ra một file tạm), cuối cùng nối các đoạn bằng FFmpeg concat. Tạm dừng/tiếp tục không mất tiến độ.
- `cli.py` với các lệnh:
  - `ecoup run -i in.mp4 -o out.mp4 [--config config.yaml] [--scale 2] [--resume]`
  - `ecoup doctor`
  - `ecoup benchmark -i sample.mp4` (đo fps, VRAM, nhiệt trung bình, khung bỏ qua)
- Thanh tiến trình, ETA, in nhiệt độ/công suất hiện tại.

**Nghiệm thu**
- Job đầy đủ cho video mẫu 1–2 phút chạy xong, có âm thanh, đồng bộ.
- Ngắt giữa chừng rồi `--resume` cho ra video hoàn chỉnh, không mất/lặp khung.
- `test_pipeline_smoke.py` pass (dùng model giả/identity để test nhanh không cần GPU).

### Phase 6 — GUI PyQt (tùy chọn, làm sau cùng)
**Công việc**
- `gui/main_window.py`: chọn file vào/ra, chọn model, scale, preset "Tiết kiệm / Cân bằng / Chất lượng".
- Hiển thị realtime: tiến độ, fps, nhiệt độ GPU, công suất, VRAM, số khung bỏ qua; nút Tạm dừng / Tiếp tục / Huỷ.
- Chạy pipeline trong `QThread`/worker, giao tiếp bằng signal/slot; **không** chạy nặng trên UI thread.
- Preset gợi ý:
  - *Tiết kiệm*: tile 192–256, `temp_soft_c=65`, `temp_hard_c=75`, dedup bật, target_scale 2.
  - *Cân bằng*: tile 256–400, `temp_soft_c=70`, `temp_hard_c=80`.
  - *Chất lượng*: tile lớn hơn, dedup tắt, model nặng hơn.

**Nghiệm thu**
- GUI chạy được job hoàn chỉnh, nút tạm dừng/tiếp tục hoạt động, UI không đơ.

### Phase 7 — Benchmark, tài liệu, đóng gói
**Công việc**
- `scripts/benchmark.py`: so sánh các cấu hình (tile size, FP16 vs FP32, ONNX CUDA vs TensorRT, có/không dedup) trên cùng một clip; xuất bảng: fps, VRAM đỉnh, nhiệt độ trung bình/đỉnh, công suất trung bình.
- Viết `README.md`: cài đặt, yêu cầu (FFmpeg, driver NVIDIA), cách export ONNX, ví dụ lệnh, mẹo giảm nhiệt (power limit, undervolt qua MSI Afterburner), khắc phục sự cố.
- Đóng gói bằng `pyproject.toml` (cài bằng `pip install -e .`).

**Nghiệm thu**
- Người dùng mới làm theo README chạy được ví dụ đầu tiên.
- Có bảng benchmark trong `docs/benchmark.md`.

## 7. Quy ước code

- Type hints đầy đủ, docstring ngắn cho hàm public.
- Không hard-code đường dẫn; dùng `pathlib`.
- Bắt và xử lý lỗi rõ ràng (thiếu FFmpeg, thiếu model, hết VRAM → tự giảm tile size và thử lại, ghi log).
- Xử lý **OOM VRAM**: nếu gặp lỗi hết bộ nhớ, tự giảm `tile_size` một bậc (ví dụ 256 → 192 → 128) và retry, tối đa 3 lần.
- Log rõ ràng, mức INFO mặc định, DEBUG bật bằng `--verbose`.
- Mỗi module quan trọng có unit test; phần cần GPU thật có thể đánh dấu `@pytest.mark.gpu` và bỏ qua mặc định.

## 8. Rủi ro & cách xử lý

| Rủi ro | Giải pháp |
|---|---|
| Đường nối tile lộ ra | Overlap + blend, kiểm tra bằng test so sánh với không-tile |
| Dedup làm video bị "đứng" | Ngưỡng bảo thủ, reset khi đổi cảnh, cho tắt được |
| NVENC không có / lỗi | Tự fallback `libx264`, báo cảnh báo |
| Không có quyền admin để đặt power limit | Chỉ cảnh báo, hướng dẫn thủ công, governor phần mềm vẫn hoạt động |
| TensorRT khó cài trên Windows | Mặc định CUDA EP; TensorRT là tuỳ chọn |
| Audio lệch / mất | Mux audio từ nguồn gốc bằng FFmpeg, test đồng bộ |
| Hết RAM do hàng đợi lớn | Giới hạn `maxsize` hàng đợi nhỏ (2–4 khung) |
| Bản quyền model | Ghi rõ license từng model trong `models/README.md` |

## 9. Định nghĩa "Hoàn thành" (Definition of Done)

- [ ] `ecoup run` upscale được video mẫu 720p → 1080p có âm thanh, đồng bộ.
- [ ] Nhiệt độ GPU được giữ dưới ngưỡng cấu hình nhờ ThermalGovernor.
- [ ] Hỗ trợ tạm dừng và `--resume` không mất tiến độ.
- [ ] Giảm tải đo được nhờ dedup + tile + FP16 (có số liệu benchmark).
- [ ] Toàn bộ test (không cần GPU) pass.
- [ ] README đủ để người khác cài và chạy.

## 10. Thứ tự ưu tiên khi thiếu thời gian

1. Phase 0 → 1 → 2 → 5 (bản chạy được tối thiểu, chưa có tối ưu).
2. Phase 4 (kiểm soát nhiệt, giá trị cốt lõi của dự án).
3. Phase 3 (giảm số khung xử lý).
4. Phase 7 (benchmark + tài liệu).
5. Phase 6 (GUI) làm cuối.

## 11. Thông tin cần xác nhận với người dùng trước khi tối ưu sâu

Nếu agent cần chọn model/cấu hình cụ thể, hãy hỏi hoặc đọc từ `ecoup doctor`:
- Loại GPU và dung lượng VRAM.
- Loại video chủ yếu (anime / phim / quay thật).
- Độ phân giải đầu vào và đầu ra mong muốn.
- Hệ điều hành (Windows/Linux) và có quyền admin hay không.
