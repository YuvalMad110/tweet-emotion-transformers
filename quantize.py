"""
quantize.py - Export and quantize a trained model to ONNX format.
Supports INT8 dynamic and static quantization with GPU inference via ONNX Runtime.
Saves quantized model to outputs/{experiment}/compressions/onnx_quantized/

Requires: pip install optimum[onnxruntime-gpu]

Example usage:
    python quantize.py --experiment DEBERTA_LARGE_09-01-54
    python quantize.py --experiment DEBERTA_LARGE_09-01-54 --quantization static
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from models import ModelType, EmotionClassifier, NUM_LABELS
from utils.utils import get_project_root
from data.dataset import load_csv
from transformers import AutoTokenizer

import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader


class ONNXCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for ONNX static quantization."""

    def __init__(self, calibration_data: list[dict]):
        self.data = calibration_data
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        sample = self.data[self.index]
        self.index += 1
        return sample

    def rewind(self):
        self.index = 0


def export_to_onnx(model: EmotionClassifier, tokenizer, output_path: Path, max_length: int = 128):
    """Export PyTorch model to ONNX format."""
    model.eval()

    # Create dummy input
    dummy_text = "This is a sample text for export."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {output_path}")


def prepare_calibration_data(
    texts: list[str], tokenizer, max_length: int = 128
) -> list[dict]:
    """Prepare calibration data for static quantization."""
    calibration_data = []
    for text in texts:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        calibration_data.append({
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        })
    return calibration_data


def get_onnx_model_size_mb(model_path: Path) -> float:
    """Get ONNX model file size in MB."""
    return model_path.stat().st_size / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Export and quantize model to ONNX")
    parser.add_argument("--experiment", type=str, default='DEBERTA_LARGE_09-01-54',
                        help="Experiment folder name")
    parser.add_argument("--quantization", type=str, default="dynamic",
                        choices=["dynamic", "static", "none"],
                        help="Quantization type: dynamic (faster), static (better accuracy), none (fp32)")
    parser.add_argument("--train_path", type=str, default="/home/yuvalmad/datasets/tweets/train.csv",
                        help="Path to training CSV for calibration (static quantization only)")
    parser.add_argument("--num_calibration_samples", type=int, default=128,
                        help="Number of samples for static quantization calibration")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    args = parser.parse_args()

    project_root = Path(get_project_root())
    experiment_dir = project_root / "outputs" / args.experiment
    checkpoint_path = experiment_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint info
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = ModelType[checkpoint["model_type"]]
    config = checkpoint.get("config", {})

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_type.value)

    # Create and load model
    print(f"Loading model: {model_type.value}")
    model = EmotionClassifier(
        model_type=model_type,
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Create output directory
    quant_suffix = f"onnx_{args.quantization}" if args.quantization != "none" else "onnx_fp32"
    output_dir = experiment_dir / "compressions" / quant_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    onnx_fp32_path = output_dir / "model.onnx"
    print("Exporting to ONNX...")
    export_to_onnx(model, tokenizer, onnx_fp32_path, args.max_length)
    fp32_size = get_onnx_model_size_mb(onnx_fp32_path)
    print(f"FP32 ONNX model size: {fp32_size:.2f} MB")

    # Apply quantization
    final_model_path = onnx_fp32_path
    quantized_size = fp32_size

    if args.quantization == "dynamic":
        print("Applying dynamic INT8 quantization...")
        quantized_path = output_dir / "model_quantized.onnx"
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        final_model_path = quantized_path
        quantized_size = get_onnx_model_size_mb(quantized_path)
        # Remove fp32 model to save space
        onnx_fp32_path.unlink()
        final_model_path.rename(onnx_fp32_path)
        final_model_path = onnx_fp32_path
        print(f"Quantized model size: {quantized_size:.2f} MB")

    elif args.quantization == "static":
        print(f"Loading calibration data from {args.train_path}...")
        texts, _ = load_csv(args.train_path, verbose=False)
        calibration_texts = texts[:args.num_calibration_samples]
        print(f"Using {len(calibration_texts)} samples for calibration")

        calibration_data = prepare_calibration_data(calibration_texts, tokenizer, args.max_length)
        calibration_reader = ONNXCalibrationDataReader(calibration_data)

        print("Applying static INT8 quantization...")
        quantized_path = output_dir / "model_quantized.onnx"
        quantize_static(
            model_input=str(onnx_fp32_path),
            model_output=str(quantized_path),
            calibration_data_reader=calibration_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        final_model_path = quantized_path
        quantized_size = get_onnx_model_size_mb(quantized_path)
        # Remove fp32 model to save space
        onnx_fp32_path.unlink()
        final_model_path.rename(onnx_fp32_path)
        final_model_path = onnx_fp32_path
        print(f"Quantized model size: {quantized_size:.2f} MB")

    compression_ratio = fp32_size / quantized_size if quantized_size > 0 else 1.0

    # Save metadata
    metadata = {
        "compression_method": f"onnx_{args.quantization}",
        "quantization_type": args.quantization,
        "original_model_type": checkpoint["model_type"],
        "fp32_size_mb": round(fp32_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "compression_ratio": round(compression_ratio, 2),
        "max_length": args.max_length,
        "original_parameters": config.get("num_parameters", "N/A"),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save config for inference
    torch.save({
        "model_type": checkpoint["model_type"],
        "config": config,
        "max_length": args.max_length,
    }, output_dir / "config.pt")

    print(f"\nDone! Saved to: {output_dir}")
    print(f"  Original size: {fp32_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"\nTo run inference with ONNX Runtime (GPU):")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{final_model_path}', providers=['CUDAExecutionProvider'])")


if __name__ == "__main__":
    main()