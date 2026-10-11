"""Script to export PyTorch super-resolution model weights (.pth) to ONNX with dynamic axes."""

import argparse
import sys
from pathlib import Path


def export_model_to_onnx(pth_path: str, onnx_path: str, scale: int = 4, fp16: bool = False) -> None:
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is required to export .pth models to ONNX.")
        print("Please install torch: pip install torch")
        sys.exit(1)

    try:
        from spandrel import ModelLoader
        print(f"Loading weights with Spandrel: {pth_path}")
        model = ModelLoader().load_from_file(pth_path)
        net = model.model
    except ImportError:
        print("[WARNING] Spandrel not installed. Attempting standard torch.load...")
        net = torch.load(pth_path, map_location="cpu")
        if isinstance(net, dict) and "params_ema" in net:
            net = net["params_ema"]

    net.eval()
    if fp16:
        net = net.half()
        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float16)
    else:
        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print(f"Exporting to ONNX: {onnx_path} (FP16={fp16}) with dynamic axes...")
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(f"Successfully exported ONNX model to {onnx_path}!")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch (.pth) to ONNX")
    parser.add_argument("-i", "--input", required=True, help="Input .pth weights path")
    parser.add_argument("-o", "--output", required=True, help="Output .onnx path")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Scale factor")
    parser.add_argument("--fp16", action="store_true", help="Export as FP16")
    args = parser.parse_args()

    export_model_to_onnx(args.input, args.output, args.scale, args.fp16)


if __name__ == "__main__":
    main()
