# KẾ HOẠCH NÂNG CẤP: Pipeline làm nét video (SeedVR2) — Đợt 2

> Tài liệu dành cho agent (Antigravity) thực thi. Làm tuần tự theo Phase; hoàn thành **Tiêu chí nghiệm thu** của Phase trước khi sang Phase sau. Sau mỗi Phase: chạy `pytest`, rồi commit.
> **Phạm vi:** độ tin cậy, tốc độ và chất lượng đầu vào cho engine SeedVR2 trên Colab. 

---

## 1. Bối cảnh và những gì đã quan sát được trong mã

Dự án `ecoup` (`src/ecoup/`, `colab_app.py`, `gui/`). Engine SeedVR2-3B nằm ở `src/ecoup/engine/seedvr2_engine.py`, chạy qua Colab UI (`colab_app.py`, Gradio).

**Nguồn video mới của người dùng:** 720×1280 (dọc, 9:16), người thật, bị nén mạng xã hội (mờ, có thể có block/nhiễu muỗi). Trước đó thử trên bản 406×720 cho kết quả mờ, nhiễu hạt và nét ngẫu nhiên.

**Số liệu hiệu năng người dùng báo (406×720, ~1700 khung):** Tesla T4 mất khoảng 3–4 giờ; L4 24GB mất khoảng 30 phút.

### Phát hiện cụ thể (cần xác minh bằng đo đạc, không coi là chắc chắn)
1. **Cửa sổ thời gian thực tế của SeedVR2 bị cứng ở 5 khung**, bất kể `clip_length` trong pipeline.
   - `SeedVR2UpscaleEngine.upscale_clip()`: nếu `len(frames) > 5` thì gọi `_upscale_long_clip()`, hàm này cắt thành các khối 5 khung (`sub_len = 5`, `sub_step = 4`) và gọi lại `upscale_clip()` cho từng khối, trộn khung chung bằng `cv2.addWeighted(..., 0.5, 0.5)`.
   - Hệ quả: tăng `clip_length` ở `colab_app.py` (đang là 5, overlap 1) **không** tăng ngữ cảnh thời gian cho model; ngữ cảnh luôn chỉ 5 khung. Mô hình video bị dùng gần như từng khung riêng lẻ.
2. **Offload GPU↔CPU lặp lại cho mỗi khối 5 khung:** VAE → GPU → encode → CPU; DiT → GPU → sampling → CPU; VAE → GPU → decode → CPU. Với ~1700 khung, khoảng 425 lần chuyển trọng số DiT 3B qua PCIe. Không có nhánh nào bỏ offload khi VRAM dư (trên L4 24GB vẫn offload).
3. **Toàn bộ dùng `bfloat16`** (autocast, VAE, DiT). T4 (Turing) không có bf16 phần cứng nên rất có thể chậm; L4 (Ada) có. Chưa kiểm chứng nguyên nhân chính.
4. **Nhiễu lấy mẫu không có seed — ĐÃ VÁ MỘT PHẦN (commit `5798fc4`):** engine hiện dùng `torch.Generator` với seed cứng `42` cho mọi khối. Còn thiếu: seed cấu hình được, ghi vào log/JSON, và chưa có A/B đo lợi ích (xem Phase 1.2).
5. **Checkpoint bị tắt trên Colab:** `colab_app.py` đặt `CheckpointConfig(enabled=False)`, dù cơ chế (`checkpoint.py`, segment trong `pipeline.py`) đã có. Phiên Colab bị ngắt = mất toàn bộ tiến độ.
6. **Khử nhiễu hiện tại yếu:** chỉ có `cv2.bilateralFilter(d=5)` trên từng khung (`pipeline.py::_preprocess_frame`), trộn lại với ảnh gốc theo tỉ lệ thanh trượt; không có lọc theo thời gian, không có khử block nén. Thanh trượt **có hoạt động** (đã truy vết `denoise_sl` → `PreProcessConfig` → `_preprocess_frame`), nhưng hiệu quả thấp.
7. **Cấu hình cứng 3B:** đường dẫn `seedvr2_ema_3b.pth`, `configs_3b/main.yaml`, repo `ByteDance-Seed/SeedVR2-3B`.
8. **Độ phân giải xử lý nội bộ của model bị cứng ở ~720×1280 (diện tích), KHÔNG phụ thuộc `target_scale`** (`upscale_clip()`: `res_h, res_w = 1280, 720` rồi `NaResize(resolution=sqrt(res_h*res_w))`). Sau đó kết quả được `cv2.resize(..., LANCZOS4)` lên kích thước đích. Hệ quả:
   - Nguồn 720×1280: model chạy ở đúng độ phân giải gốc (chỉ phục chế, không "siêu phân giải"), rồi phóng 1.5x/2x bằng Lanczos thông thường. Chọn 2x trong UI gần như chỉ phóng nội suy.
   - Nguồn 406×720: model nhận ảnh đã phóng ~1.77x, phần còn lại cũng do Lanczos.
   - Đây là điểm ảnh hưởng chất lượng lớn nhất và **chưa có trong kế hoạch cũ** → xử lý ở Phase 4.0.
9. **Backend attention trên T4:** FlashAttention-2 không hỗ trợ Turing (T4); cần xác minh ở Phase 0 xem SeedVR đang rơi về SDPA/xformers hay đường chậm hơn.

---

## 2. Mục tiêu

- Chạy ổn định và **bền** trên Colab (ngắt phiên không mất tiến độ).
- **Nhanh hơn rõ rệt** trên GPU nhiều VRAM, không phá vỡ khả năng chạy trên T4 16GB.
- **Chất lượng đầu vào sạch hơn** (khử nén có lọc theo thời gian) và kết quả ổn định hơn giữa các khối.
- Hỗ trợ đích **tùy chọn 1.5x (1080×1920)** và 2x (1440×2560) từ nguồn 720×1280.
- Có công cụ **đo và xem trước nhanh** để tránh phải chạy cả video chỉ để thử tham số.

## 3. Ràng buộc

- Python 3.10+; giữ phong cách và cấu trúc hiện có. Không phá vỡ engine ONNX/CCSR và các test hiện có.
- Mọi tham số mới cấu hình được qua dataclass trong `config.py` (kèm `validate()`) và hiển thị trong Colab UI khi hợp lý; không hard-code.
- **Mọi con số về VRAM, tốc độ, ngưỡng đều phải đo thực tế**, ghi vào `docs/benchmark.md`. Không đoán.
- Không commit video, trọng số model hoặc file lớn.
- Không tải model từ nguồn không rõ; ghi nguồn và checksum vào `models/README.md`.

---

## Phase 0: Đo đạc nền (bắt buộc làm trước)

**Việc cần làm**
1. Thêm module ghi thời gian từng bước trong `seedvr2_engine.py` (ví dụ context manager `timed("tên_bước")` gom vào dict, in ra khi kết thúc): VAE encode, chuyển DiT lên GPU, DiT sampling, chuyển DiT về CPU, VAE decode, chuyển VAE. Bật bằng cờ cấu hình, mặc định tắt (hoặc ghi mức DEBUG).
2. Ghi thêm: VRAM đỉnh (`torch.cuda.max_memory_allocated`), tên GPU, dtype đang dùng.
3. Tạo `scripts/profile_seedvr2.py`: nhận một clip ngắn (30–40 khung) và xuất bảng thời gian từng bước. Viết kết quả vào `docs/benchmark.md` (mục "SeedVR2 profile").
4. Chạy trên T4 và L4 (nếu có quyền) và ghi kết quả; nếu không có quyền chạy Colab, **ghi rõ chưa đo** và để lại script cho người dùng.

**Nghiệm thu:** có script profile chạy được; có (hoặc có hướng dẫn rõ để người dùng tạo) bảng thời gian từng bước trên GPU thật.

---

## Phase 1: Độ tin cậy trên Colab (ưu tiên cao nhất)

**1.1 Bật checkpoint trên Colab**
- Trong `colab_app.py`, bật `CheckpointConfig(enabled=True)` với thư mục lưu trên **Google Drive hoặc thư mục bền vững** (tùy chọn, có mặc định an toàn).
- Kiểm tra `pipeline.py` + `checkpoint.py` hoạt động với engine clip (segment, ghép lại cuối, âm thanh ghép ở bước concat). Sửa nếu có lỗi (đặc biệt đồng bộ cross-fade giữa các khối khi resume).
- Thêm nút/tùy chọn "Tiếp tục job dở" trong UI.

**1.2 Seed cố định cho nhiễu lấy mẫu**
- Engine đã dùng `torch.Generator` seed cứng `42` (commit `5798fc4`). Việc còn lại: đưa thành `seed: Optional[int]` trong cấu hình (mặc định 42), tùy chọn offset theo chỉ số khối, ghi seed vào log và JSON cạnh đầu ra.
- Cho phép đặt `None` để ngẫu nhiên.
- Lưu ý: cùng seed cho mọi khối giúp ổn định chi tiết nhưng có thể lộ họa tiết nhiễu cố định; A/B thử "cùng seed" vs "offset theo khối".
- **Không** khẳng định lợi ích chất lượng trước khi đo: so sánh có/không seed trên cùng clip và ghi vào `docs/eval_report.md`.

**Nghiệm thu:** chạy thử một video 300+ khung, ngắt giữa chừng, tiếp tục và kiểm tra: số khung đầu ra = đầu vào, âm thanh khớp, không giật ở chỗ nối. `pytest` xanh.

---

## Phase 2: Tốc độ và VRAM thích ứng

**2.1 Chọn chế độ bộ nhớ khi `load()`**
- Đọc VRAM trống (`torch.cuda.mem_get_info`) và quyết định:
  - `resident`: giữ DiT và VAE trên GPU suốt, không offload (khi VRAM dư).
  - `offload`: giữ hành vi hiện tại (T4 16GB).
- Thêm cấu hình `memory_mode: auto | resident | offload`. `auto` dùng ngưỡng theo **số đo ở Phase 0** (không đặt số tùy ý); có log nói rõ chế độ đã chọn và vì sao.
- Với `resident`, bỏ các lệnh `.to("cpu")` / `.to(device)` lặp trong vòng lặp khối; vẫn giữ `_clear_gpu()` ở mức hợp lý.

**2.2 Tăng cửa sổ thời gian thật của model (sửa phát hiện số 1)**
- Làm cho kích thước khối thời gian (`sub_len`, `sub_step`) **cấu hình được** thay vì cứng 5/4, theo dạng `4k+1` (5, 9, 13, 17, 21...). Đây là thay đổi quan trọng cho cả chất lượng lẫn số lần offload.
- Tối ưu cách trộn khung chung giữa các khối (hiện cross-fade 0.5/0.5 một khung); đề xuất trộn trọng số tuyến tính trên nhiều khung gối đầu nếu đo thấy giật.
- Với `memory_mode=offload` trên T4, tăng `sub_len` làm tăng VRAM; cần có **cơ chế tự hạ** về khối nhỏ hơn khi gặp OOM (bắt `torch.cuda.OutOfMemoryError`, giải phóng bộ nhớ, thử lại với `sub_len` nhỏ hơn, ghi log).
- Cập nhật `colab_app.py`: `clip_length`/`clip_overlap` truyền xuống phải khớp ý nghĩa mới (làm rõ chú thích hai khái niệm: đoạn của pipeline và cửa sổ của model).


**2.3 Thống nhất việc chia khối giữa pipeline và engine (tránh chia hai lần)**
- Hiện `pipeline.py` chia clip (`clip_length`, `clip_overlap`, cross-fade tuyến tính) rồi `upscale_clip()` lại chia tiếp thành khối 5 khung với cross-fade 0.5/0.5. Hai lớp chồng nhau gây trộn khung hai lần và khó suy luận.
- Phương án: để **engine sở hữu** việc chia khối/trộn (pipeline đưa cả đoạn dài, hoặc `clip_length` = `sub_len`, `clip_overlap` = 1), và xóa lớp trộn thừa. Giữ test số khung đầu ra = đầu vào.

**2.4 Thử fp16 trên T4 (thử nghiệm, có kiểm tra chất lượng) — LÀM SỚM, ngay sau Phase 0**
- Lý do ưu tiên: T4 không có bf16 phần cứng, đây là nghi phạm số 1 cho chênh lệch ~6–8 lần so với L4 (cùng với offload). Đo sớm để biết có đáng đầu tư các mục sau không.
- Thêm tùy chọn `precision: bf16 | fp16`. Chỉ khuyến nghị fp16 trên T4 nếu A/B cho thấy nhanh hơn rõ **và** không tràn số/NaN/artifact (kiểm tra NaN/Inf ở đầu ra DiT và VAE, có fallback về bf16). Ghi kết quả vào báo cáo; nếu chất lượng giảm, giữ bf16.
- Nếu fp16 không dùng được trên T4, ghi nhận kết luận "T4 chỉ phù hợp clip ngắn; video dài dùng L4/A100".

**Nghiệm thu:** `docs/benchmark.md` có bảng trước/sau trên T4 và L4 (hoặc ghi rõ chưa đo): thời gian/khung, VRAM đỉnh, chế độ bộ nhớ, `sub_len`. Kết quả hình ảnh không tệ hơn baseline (so cạnh nhau trên cùng clip). `pytest` xanh, có test với engine giả cho logic chọn chế độ và tự hạ khi OOM.

---

## Phase 3: Khử nhiễu và khử nén đầu vào

**3.1 Kiến trúc bộ lọc tiền xử lý**
- Thay `_preprocess_frame` (từng khung) bằng một **bộ lọc theo đoạn** (nhận danh sách khung để có thể lọc theo thời gian), giữ tương thích ngược với `bilateral` hiện có.
- Thêm cấu hình `pre_process.method: none | bilateral | hqdn3d | nlmeans | deblock+hqdn3d` và các tham số cường độ.
- **Khuyến nghị kiến trúc:** các bộ lọc theo thời gian (hqdn3d, nlmeans, deblock) nên chạy như **một bước tiền xử lý FFmpeg trên cả video → file trung gian**, rồi mới đưa vào pipeline. Nhanh, không phải cài lại lọc thời gian bằng OpenCV, không dính cross-fade/overlap của khối, và hợp với checkpoint/resume. Chỉ giữ bilateral theo khung trong Python cho tương thích ngược.
- Cài đặt gợi ý:
  - `hqdn3d`: FFmpeg filter (kiểm tra FFmpeg trên Colab có sẵn; cần file trung gian chất lượng cao, ví dụ CRF thấp, tránh nén lại gây thêm block).
  - `nlmeans`: `cv2.fastNlMeansDenoisingColoredMulti` (nhiều khung) hoặc FFmpeg `nlmeans`; cảnh báo về tốc độ.
  - `deblock`: FFmpeg `deblock` hoặc `pp=ha/va/dr` (xác minh bộ lọc có trong bản FFmpeg đang dùng).
- Bảo đảm bộ lọc không phá vỡ cơ chế cross-fade/overlap của khối.

**3.2 Xem trước trước-sau**
- Thêm trong Colab UI chức năng chỉ chạy tiền xử lý trên 1–3 khung đại diện và hiển thị ảnh trước/sau (có phóng to vùng mặt hoặc vùng chọn). Không chạy model AI ở bước này.

**3.3 Thử nghiệm ma trận nhỏ**
- Chuẩn bị clip 30–40 khung từ nguồn 720×1280. Chạy: (none / hqdn3d vừa / deblock+hqdn3d / nlmeans) × (cửa sổ `sub_len` 5 / 9 hoặc 13 nếu đủ VRAM). Lưu kết quả ở `eval/results/` và ghi nhận xét (mặt, da, nhấp nháy) vào `docs/eval_report.md`.
- Chọn **giá trị mặc định** theo kết quả; bỏ phương pháp nào không có ích. **Không** đặt mặc định cực cao: đã quan sát bilateral 75% chỉ cải thiện rất nhỏ, và mức quá mạnh làm da bệt.

**Nghiệm thu:** có ít nhất 3 phương pháp lựa chọn trong cấu hình và UI; có xem trước trước/sau; có số liệu A/B và mặc định được chọn có lý do. `pytest` xanh.

---

## Phase 4: Độ phân giải đích và thông tin nguồn

0. **Độ phân giải xử lý nội bộ cấu hình được (sửa phát hiện số 8) — QUAN TRỌNG:**
   - Thêm `inference.model_resolution` (ví dụ `native | match_target | số cạnh dài`), thay cho `1280, 720` cứng. Với `match_target`, model xử lý ở độ phân giải gần kích thước đích (1.5x/2x) để thực sự tạo chi tiết thay vì để Lanczos phóng.
   - Chi phí tăng gần tỉ lệ diện tích (1.5x ≈ 2.25 lần, 2x ≈ 4 lần token) → phải gắn với `memory_mode`, tự hạ `sub_len`, và đo VRAM/thời gian ở Phase 0/2. Trên T4 có thể chỉ khả thi với `native` hoặc 1.5x; ghi rõ giới hạn.
   - A/B trên cùng clip: `native + Lanczos 2x` vs `match_target`. Nếu `match_target` không tốt hơn rõ, giữ `native` làm mặc định và ghi vào báo cáo.
1. **Đích tùy chọn:** cho phép `target_scale` 1.5 và 2.0 (1280×720 → 1080×1920 hoặc 1440×2560 cho nguồn dọc 720×1280); đảm bảo kích thước đầu ra chia hết theo yêu cầu của encoder (số chẵn) và của model (`DivisibleCrop((16,16))` — kiểm tra không cắt mất biên không mong muốn; nếu có, pad rồi crop về đúng kích thước).
2. **Hiển thị thông tin nguồn** bằng `ffprobe` (độ phân giải, codec, bitrate, fps, thời lượng) ngay khi chọn file, kèm cảnh báo khi nguồn nhỏ hoặc bitrate thấp.
3. **Kiểm tra mã hóa đầu ra:** xác minh `codec: auto` thực sự chọn gì trên T4/L4 (NVENC có sẵn không); kiểm tra CRF/CQ phù hợp video dọc; ghi vào README.
4. **Chế độ xem thử 30–40 khung** cho toàn pipeline (khử nhiễu → model → hậu xử lý), xuất ra video ngắn để xem trước khi chạy cả video.

**Nghiệm thu:** chạy thành công ở cả 1.5x và 2x trên clip thử; thông tin nguồn hiển thị đúng; chế độ xem thử xuất được video ngắn.

---

## Phase 5 (tùy chọn): Hỗ trợ SeedVR2-7B

Chỉ làm khi người dùng xác nhận có GPU 24GB trở lên **và** các Phase trước chưa cho chất lượng đủ.
- Tổng quát hóa engine: đường dẫn checkpoint, file cấu hình (`configs_7b` nếu có trong repo gốc), tên repo Hugging Face; chọn 3B/7B qua cấu hình.
- Kiểm tra VRAM thực tế và logic `memory_mode` cho 7B; không dùng quantize mạnh nếu làm giảm chất lượng (đo).
- Ghi nguồn, giấy phép, checksum vào `models/README.md`.

**Nghiệm thu:** chạy được 7B trên clip thử; có so sánh 3B và 7B (chất lượng, thời gian, VRAM) trong `docs/eval_report.md`.

---

## Phase 6: Kiểm thử, tài liệu, chốt

1. **Test (không gọi model thật):** engine giả để kiểm tra: chọn `memory_mode`, tự hạ khi OOM, chia khối `4k+1`, cross-fade và số khung đầu ra, seed tái lập, checkpoint/resume, bộ lọc tiền xử lý (kích thước/kiểu dữ liệu đầu ra đúng).
2. Cập nhật `README.md`: chế độ bộ nhớ, tham số mới, khuyến nghị cho T4 và L4/A100, giới hạn đã biết.
3. Ghi lại **cấu hình đã dùng** (kèm seed) cạnh file đầu ra (`.json`) để lặp lại chính xác.
4. Chạy một video dài 2–3 phút trên Colab: đo tổng thời gian, VRAM đỉnh, hành vi ngắt/tiếp tục; cập nhật `docs/benchmark.md`.

**Nghiệm thu:** README và báo cáo đầy đủ; người dùng xác nhận kết quả.

---

## Phase 7 (tùy chọn): Xử lý theo vùng / khôi phục mặt

Chỉ làm khi người dùng xác nhận và sau khi Phase 4.0 đã đo xong, vì với nguồn toàn thân mặt nhỏ (~50px ở 406×720) SeedVR2 vẫn "tưởng tượng" môi/mũi/mắt.
- **7.1 Chia vùng trên/dưới:** crop 2 vùng có chồng lấn (ví dụ 406×406 ở y=0 và y=314 cho nguồn 406×720), upscale riêng, ghép bằng mặt nạ gradient ở vùng chồng. Có tùy chọn UI; cảnh báo tốn gấp đôi thời gian và có thể lộ đường ghép.
- **7.2 Khôi phục mặt (CodeFormer/GFPGAN):** phát hiện mặt từng khung, khôi phục, dán lại có hòa trộn, ổn định theo thời gian (làm mượt hộp mặt). Ghi nguồn/giấy phép/checksum vào `models/README.md`.
- **Nghiệm thu:** so sánh cạnh nhau với đường chuẩn trên clip có người toàn thân; không lộ đường ghép, không nhấp nháy mặt.

---

## 4. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Tăng cửa sổ thời gian gây OOM trên T4 | Cơ chế tự hạ `sub_len` khi OOM (Phase 2.2) |
| fp16 gây tràn số hoặc artifact | Chỉ bật mặc định nếu A/B sạch; giữ bf16 làm mặc định |
| Khử nhiễu quá mạnh làm da bệt | Mặc định vừa phải theo A/B; có xem trước trước-sau |
| Tiling lộ đường nối ở DiT/VAE | Overlap + blend viền; kiểm tra bằng mắt và so sánh |
| Colab ngắt phiên | Checkpoint trên Drive (Phase 1); kiểm thử resume |
| Nguồn nén nặng giới hạn chất lượng | Chấp nhận: mục tiêu "sạch và ổn định", không hứa "như quay gốc" |
| Kết quả đo khác giả định trong tài liệu này | Tin số đo, cập nhật kế hoạch; báo người dùng |

## 5. Quy tắc làm việc cho agent

- Làm đúng thứ tự Phase; **Phase 0 trước**. Dừng và hỏi người dùng khi cần chọn mặc định ảnh hưởng chất lượng (cuối Phase 3) hoặc có GPU để đo thật.
- Các "phát hiện" ở mục 1 là suy luận từ đọc mã: **xác minh bằng đo/chạy thử trước khi dựa vào**, và báo cáo nếu sai.
- Không thêm tính năng ngoài phạm vi (watermark, ảnh, phụ đề, nội suy khung). Phục hồi mặt chỉ làm ở Phase 7 tùy chọn, và chỉ khi được người dùng đồng ý.
- **Thứ tự thực hiện đề xuất:** Phase 0 (đo, kèm xác minh attention backend) → Phase 2.4 (thử fp16, nhanh, đo sớm) → Phase 1 (checkpoint, seed; độc lập GPU, có thể làm song song) → Phase 4.0 (độ phân giải nội bộ) → Phase 2.1–2.3 → Phase 3 → Phase 6. Agent không tự chạy được Colab: các bước đo cần người dùng chạy `scripts/profile_seedvr2.py` và dán kết quả; nếu chưa có, ghi "chưa đo" và không đặt ngưỡng tùy ý.
- Sau mỗi Phase, báo cáo ngắn: đã làm gì, số liệu đo, vấn đề còn lại.
- Không commit video, trọng số model hoặc file lớn.
