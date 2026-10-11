# Models Directory

Thư mục lưu trữ các mô hình AI ONNX cho Eco Video Upscaler.

## 1. Nguồn mô hình đề xuất (Lightweight / Low VRAM)

| Model Name | Kiến trúc | Tác vụ | File ONNX đề xuất | Nguồn gốc |
|---|---|---|---|---|
| `realesr-animevideov3` | Compact (SRVGGNet) | Video Anime, hoạt hình, 2D | `realesr-animevideov3.onnx` | [Real-ESRGAN Release](https://github.com/xinntao/Real-ESRGAN/releases) |
| `realesr-general-x4v3` | Compact | Video thực tế, người thật, tổng quát | `realesr-general-x4v3.onnx` | [Real-ESRGAN Release](https://github.com/xinntao/Real-ESRGAN/releases) |
| `SPAN-x2` | SPAN | Tốc độ cao, siêu nhẹ | `span_x2.onnx` | [ChaiNNer / OpenModelDB](https://openmodeldb.info/) |

## 2. Lưu ý
- Các file mô hình `.onnx` và `.pth` được bỏ qua bởi `.gitignore` để không làm phình dung lượng repository git.
- Bạn có thể xuất mô hình từ PyTorch weights `.pth` sang `.onnx` bằng script `scripts/export_onnx.py`.
