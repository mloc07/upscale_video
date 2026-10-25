# KẾ HOẠCH: Module phụ đề tự động (Whisper → Gemini → ghép vào video)

> Tài liệu dành cho agent (Antigravity) thực thi. Làm tuần tự theo Phase; hoàn thành **Tiêu chí nghiệm thu** của Phase trước khi sang Phase sau. Sau mỗi Phase: chạy `pytest`, rồi commit.
> **Phạm vi:** tính năng phụ đề trong một tab riêng. **Không** thay đổi thuật toán upscale/SeedVR2.

---

## 1. Mục tiêu

Quy trình **hoàn toàn tự động, một lần bấm**:

1. Tách âm thanh từ video.
2. Nhận dạng giọng nói bằng **Whisper**, tự nhận diện ngôn ngữ nguồn (Anh, Trung, Nhật, Hàn và các ngôn ngữ khác).
3. Dịch sang **tiếng Việt** bằng **Gemini** qua API (model cấu hình được; ý định ban đầu là dòng Flash Lite, **xác minh ID model chính xác trong tài liệu Gemini hiện hành**, không hard-code).
4. Xuất phụ đề `.srt` (gốc và đã dịch) và **ghép cứng (burn-in)** bản dịch vào video.

Không có bước xem trước bắt buộc, nhưng **luôn lưu các file phụ đề rời** để người dùng sửa và ghép lại khi cần.

## 2. Bối cảnh mã nguồn (đã đọc)

- Package `ecoup` ở `src/ecoup/` (`pipeline.py`, `video_io.py`, `config.py`, `cli.py`, `engine/`...). CLI: `ecoup = ecoup.cli:main`.
- `video_io.py` đã gọi FFmpeg qua `subprocess` (`VideoReader`, `VideoWriter`, `check_ffmpeg()` trong `cli.py`).
- Giao diện Colab: `colab_app.py`, một `gr.Blocks` duy nhất (chưa dùng `gr.Tab`); GUI máy tính `gui/main_window.py` (PyQt6, hiện chưa có tab).
- `requirements.txt` và `pyproject.toml` hiện **chưa** có thư viện ASR hay Google API.
- Dự án có thể sắp đổi tên repo/package; **đặt mã phụ đề độc lập tên** (đường dẫn `src/<pkg>/subtitle/`) để việc đổi tên sau này chỉ là thay import.

## 3. Ràng buộc

- Python 3.10+; giữ phong cách mã hiện có. Không phá vỡ test và chức năng upscale hiện có.
- Mọi tham số cấu hình được qua dataclass trong `config.py` (kèm `validate()`); không hard-code.
- **Khóa API không bao giờ vào repo**: đọc từ biến môi trường (ví dụ `GEMINI_API_KEY`) hoặc Colab Secrets. Không ghi khóa vào log, file cấu hình hay tên file.
- **Bộ nhớ GPU:** Whisper và SeedVR2 **không chạy cùng lúc** trên một GPU. Nạp, dùng, rồi giải phóng (`del`, `torch.cuda.empty_cache()`).
- Unit test không gọi API thật hay model thật (dùng giả/mock).
- Mọi con số hiệu năng phải **đo**, không đoán.

### Quy tắc về nội dung nhạy cảm (QUAN TRỌNG)
- Prompt dịch là prompt dịch thuật **bình thường**: dịch trung thực, giữ giọng điệu và từ thô tục, không tóm tắt hay lược bỏ.
- **Không** viết prompt kiểu jailbreak/đánh lừa nhằm vượt bộ lọc an toàn của model. Chỉ dùng cơ chế **hợp lệ của API**: tham số `safety_settings` trong phạm vi mà nhà cung cấp cho phép (đọc tài liệu hiện hành).
- Khi API **chặn hoặc từ chối** một lô: xử lý có kiểm soát (xem Phase 3.4): thử lại lô nhỏ hơn, rồi dùng backend dự phòng nếu được cấu hình; câu vẫn không dịch được thì **giữ nguyên bản gốc, đánh dấu cờ cảnh báo** và ghi vào báo cáo. Quy trình không được dừng hoặc hỏng.

## 4. Kiến trúc đề xuất

```
src/<pkg>/subtitle/
  __init__.py
  config.py        # SubtitleConfig (dataclass + validate)
  audio.py         # tách âm thanh bằng FFmpeg
  asr.py           # Whisper (faster-whisper) -> danh sách đoạn có mốc thời gian
  segments.py      # kiểu dữ liệu Segment, hợp nhất/tách đoạn, quy tắc độ dài
  translate/
    base.py        # TranslatorBackend (interface)
    gemini.py      # triển khai Gemini
    (mở rộng sau: backend LLM cục bộ)
  srt.py           # đọc/ghi .srt, tạo .ass
  burn.py          # ghép cứng bằng FFmpeg (libass)
  pipeline.py      # SubtitlePipeline: chạy tuần tự các bước, có tiến độ và hủy
  report.py        # báo cáo: ngôn ngữ phát hiện, số đoạn, số đoạn dịch lỗi/bị chặn, thời gian
```

- **Giao diện (UI):** tab riêng trong `colab_app.py` (bọc phần giao diện hiện có trong `gr.Tabs` / `gr.Tab("Upscale")`, thêm `gr.Tab("Phụ đề")`) và một tab tương ứng trong `gui/main_window.py`. UI chỉ là lớp mỏng gọi `SubtitlePipeline`; logic không nằm trong UI.
- **CLI:** thêm lệnh con, ví dụ `subtitle -i video.mp4 -o out.mp4 --lang auto --target vi`.

## 5. Phase thực hiện

### Phase 0: Chuẩn bị và xác minh môi trường
1. Kiểm tra FFmpeg trên Colab và máy cục bộ có **libass** (bộ lọc `subtitles`/`ass`) và chọn được phông hỗ trợ dấu tiếng Việt. Thêm vào `doctor` (nếu có) hoặc script kiểm tra.
2. Quyết định thư viện ASR (đề xuất `faster-whisper`; xác minh phiên bản, VRAM, tương thích CUDA trên Colab T4/L4) và thêm vào `requirements.txt` / `pyproject.toml` dưới dạng **tùy chọn** (`extras`), để không bắt người chỉ dùng upscale phải cài.
3. Xác minh trong tài liệu hiện hành: tên SDK Gemini (`google-genai` hay tên khác), ID model, giới hạn tốc độ, cách đặt `safety_settings`, định dạng phản hồi (JSON có cấu trúc nếu hỗ trợ).
4. Tạo `docs/subtitle_notes.md` ghi các kết quả xác minh trên (ngày kiểm tra, nguồn).

**Nghiệm thu:** có `docs/subtitle_notes.md`; FFmpeg libass và phông tiếng Việt kiểm tra được; thư viện thêm dưới dạng tùy chọn.

### Phase 1: Tách âm thanh và nhận dạng (ASR)
1. `audio.py`: tách âm thanh ra WAV mono 16 kHz (hoặc định dạng ASR yêu cầu) bằng FFmpeg; xử lý video không có tiếng (báo lỗi rõ ràng, không treo).
2. `asr.py`: chạy Whisper (mặc định large-v3, cấu hình được), **tự nhận diện ngôn ngữ**, bật **lọc VAD** để giảm "bịa chữ" ở đoạn im lặng/nhạc nền; trả về `Segment(start, end, text, language)`.
3. Xử lý nhiều ngôn ngữ: tùy chọn ép ngôn ngữ nguồn thủ công khi tự nhận diện sai; nếu ngôn ngữ nguồn đã là tiếng Việt thì **bỏ qua bước dịch**.
4. Giải phóng model ASR sau khi xong (kiểm tra VRAM giảm).
5. Test với ASR giả: định dạng đầu ra, xử lý rỗng, hủy giữa chừng.

**Nghiệm thu:** chạy trên một clip ngắn có tiếng cho ra `.srt` bản gốc đúng mốc thời gian; đo và ghi thời gian + VRAM đỉnh; model được giải phóng.

### Phase 2: Chuẩn hóa đoạn và xuất SRT
1. `segments.py`: hợp nhất đoạn quá ngắn, tách đoạn quá dài theo giới hạn (số ký tự mỗi dòng, số dòng, thời lượng tối thiểu/tối đa, ký tự mỗi giây). Giá trị cấu hình được, mặc định hợp lý ghi rõ trong README.
2. `srt.py`: ghi/đọc `.srt` chuẩn (kiểm tra mốc thời gian tăng, không chồng lấn, mã hóa UTF-8).
3. Test: các trường hợp biên (đoạn trống, thời gian chồng, ký tự đặc biệt, tiếng Việt có dấu).

**Nghiệm thu:** `.srt` sinh ra mở được trong trình phát phổ biến, đúng thứ tự và không chồng lấn; test xanh.

### Phase 3: Dịch bằng Gemini
1. `translate/base.py`: interface `TranslatorBackend.translate(batch: list[Segment], context) -> list[TranslatedSegment]`, để sau có thể thêm backend khác mà không sửa pipeline.
2. `translate/gemini.py`:
   - Dịch **theo lô có gối đầu** (một vài đoạn trước/sau làm ngữ cảnh), không dịch từng câu riêng lẻ.
   - Yêu cầu đầu ra có cấu trúc (JSON) và **kiểm tra số đoạn đầu ra = số đoạn đầu vào**; nếu lệch thì thử lại lô nhỏ hơn.
   - Prompt hệ thống: dịch trung thực sang tiếng Việt; **giữ nguyên số dòng/ID đoạn**; xử lý đại từ xưng hô tự nhiên, nhất quán (anh/chị/em/tôi...), cho phép nhận **ghi chú ngữ cảnh** và **bảng thuật ngữ/tên riêng** từ người dùng; giữ giọng điệu và từ thô tục; không thêm lời giải thích. (Xem quy tắc nội dung nhạy cảm ở mục 3.)
   - Cấu hình: tên model, kích thước lô, số đoạn gối đầu, nhiệt độ, thời gian chờ, số lần thử lại.
3. **Giới hạn tốc độ và lỗi mạng:** thử lại với giãn cách tăng dần (exponential backoff), tôn trọng giới hạn của API.
4. **Xử lý bị chặn/từ chối:** nhận biết phản hồi bị chặn (kiểm tra mã/lý do kết thúc theo tài liệu SDK), thử lại với lô nhỏ hơn, rồi dự phòng backend khác nếu có; đoạn cuối cùng vẫn thất bại thì giữ bản gốc kèm cờ cảnh báo, và ghi vào `report`.
5. **Cache theo lô** (khóa = băm nội dung + tham số): khi bị ngắt hoặc chạy lại không phải dịch lại phần đã xong; tiếp tục được từ lô dở.
6. Test với `TranslatorBackend` giả: lệch số dòng, thử lại, bị chặn, cache, gối đầu đúng.

**Nghiệm thu:** dịch thành công một `.srt` mẫu ở ít nhất 3 ngôn ngữ nguồn (kiểm bằng mắt, người dùng đánh giá chất lượng); kịch bản bị chặn không làm hỏng quy trình và được ghi trong báo cáo; khóa API không xuất hiện trong log.

### Phase 4: Ghép phụ đề vào video
1. `burn.py`: tạo `.ass` (hoặc dùng `.srt` với `force_style`) với phông hỗ trợ tiếng Việt, cỡ chữ tỉ lệ theo độ phân giải, viền/đổ bóng dễ đọc, vị trí chân dưới có lề an toàn (kể cả video dọc 9:16).
2. Ghép cứng bằng FFmpeg (`-vf subtitles=...`/`ass=...`), sao chép âm thanh nguyên bản, dùng cùng codec/chất lượng mã hóa đã cấu hình (tái dùng cấu hình `encode`).
3. Cẩn thận đường dẫn có ký tự đặc biệt/khoảng trắng khi đưa vào bộ lọc FFmpeg (escape đúng); kiểm tra trên cả Windows và Linux.
4. Cũng xuất `.srt`/`.ass` rời cạnh video đầu ra.

**Nghiệm thu:** video đầu ra hiển thị đúng phụ đề tiếng Việt có dấu, không mất tiếng, đúng độ dài; kiểm tra với video ngang và dọc.

### Phase 5: Pipeline end-to-end và giao diện
1. `SubtitlePipeline`: chạy các bước tuần tự, báo tiến độ từng bước, hỗ trợ **hủy**, trả `report`.
2. Tab "Phụ đề" trên Colab: chọn video, ngôn ngữ nguồn (tự động/thủ công), model Whisper, model dịch, ô ghi chú ngữ cảnh/bảng thuật ngữ, kiểu chữ, nút chạy/hủy; hiển thị tiến độ và liên kết tải `.srt` gốc, `.srt` đã dịch, video đã ghép.
3. Tab tương ứng trong PyQt (chạy ở luồng riêng, không chặn giao diện, giống tab upscale).
4. CLI lệnh con `subtitle`.
5. Công tắc "xem trước/sửa trước khi ghép" **mặc định tắt** (để dành cho đợt sau; chỉ cần chừa chỗ trong cấu hình).

**Nghiệm thu:** chạy trọn một lượt trên Colab từ một video thật tới video có phụ đề, không cần can thiệp tay; hủy giữa chừng không để lại tiến trình treo.

### Phase 6 (tùy chọn): Tích hợp với upscale
- Khi người dùng chạy cả upscale và phụ đề: **ghép phụ đề trong lần mã hóa cuối** của pipeline upscale (thêm bộ lọc phụ đề vào lệnh FFmpeg của `VideoWriter`) để tránh mã hóa hai lần. Chỉ làm khi bước này không làm phức tạp checkpoint/resume; nếu phức tạp, chấp nhận ghép ở một lượt FFmpeg riêng.
- Phụ đề phải ghép **ở độ phân giải đích, sau upscale**, không ghép trước.

### Phase 7: Kiểm thử, tài liệu, chốt
1. Test đơn vị đầy đủ cho các module (không gọi API/model thật); test tích hợp nhẹ bằng backend giả.
2. Cập nhật `README.md`: cài đặt phần phụ đề (extras), cách đặt khóa API, quyền riêng tư (lời thoại được gửi tới nhà cung cấp API), giới hạn đã biết.
3. Chạy video dài (vài phút): đo thời gian từng bước, số lần thử lại, số đoạn bị chặn; ghi vào `docs/benchmark.md`.

**Nghiệm thu:** README và báo cáo đầy đủ; người dùng xác nhận kết quả.

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Whisper bịa chữ ở đoạn im lặng/nhạc | Bật VAD; ngưỡng cấu hình được; lưu `.srt` gốc để rà |
| Dịch sai đại từ xưng hô tiếng Việt | Ngữ cảnh gối đầu, ghi chú ngữ cảnh và bảng thuật ngữ từ người dùng |
| Phản hồi API lệch số dòng | Kiểm tra, thử lại lô nhỏ hơn |
| API chặn nội dung | Xử lý có kiểm soát (Phase 3.4), không vượt bộ lọc bằng prompt đánh lừa |
| Vượt giới hạn tốc độ / mất mạng | Backoff, cache theo lô, tiếp tục được |
| Hết VRAM khi dùng cùng SeedVR2 | Chạy tuần tự, giải phóng model sau mỗi bước |
| Phụ đề burn-in không đảo ngược | Luôn xuất kèm `.srt` rời; giữ video gốc |
| Lộ khóa API | Chỉ dùng biến môi trường/Colab Secrets; không ghi vào log |
| Lời thoại riêng tư gửi ra ngoài | Ghi rõ trong README; backend cục bộ là hướng mở rộng |

## 7. Quy tắc làm việc cho agent

- Làm đúng thứ tự Phase. **Phase 0 trước:** mọi chi tiết SDK/model/API phải xác minh trong tài liệu hiện hành, không dựa vào trí nhớ.
- Không viết prompt hay mã nhằm vượt bộ lọc an toàn của model; chỉ dùng cơ chế hợp lệ của API và xử lý lỗi có kiểm soát.
- Không thêm tính năng ngoài phạm vi (xóa watermark, upscale ảnh, nội suy khung, phục hồi mặt).
- Báo cáo ngắn sau mỗi Phase: đã làm gì, số liệu đo, vấn đề còn tồn tại.
- Không commit video, khóa API, trọng số model hay file lớn.
