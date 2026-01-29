"""
quantize_fp16.py - Convert a trained model to FP16 (half precision) for GPU inference.
Saves the FP16 model to outputs/{experiment}/compressions/fp16/

FP16 provides ~2x memory reduction and faster inference on GPU with minimal accuracy loss.

Example usage:
    python quantize_fp16.py --experiment DEBERTA_LARGE_09-01-54
"""

import argparse
import json
import torch
from pathlib import Path

from models import ModelType, EmotionClassifier
from utils.utils import get_project_root


def get_model_size_mb(model) -> float:
    """Get model size in MB."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Convert model to FP16 for GPU inference")
    parser.add_argument("--experiment", type=str, default='DEBERTA_LARGE_09-01-54',
                        help="Experiment folder name")
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

    # Create and load model in FP32
    print(f"Loading model: {model_type.value}")
    model = EmotionClassifier(
        model_type=model_type,
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Get FP32 size
    fp32_size = get_model_size_mb(model)
    print(f"FP32 model size: {fp32_size:.2f} MB")

    # Convert to FP16
    print("Converting to FP16...")
    model_fp16 = model.half()
    fp16_size = get_model_size_mb(model_fp16)
    print(f"FP16 model size: {fp16_size:.2f} MB")

    # Create output directory
    output_dir = experiment_dir / "compressions" / "fp16"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FP16 model checkpoint
    print("Saving FP16 model...")
    torch.save({
        "model_type": checkpoint["model_type"],
        "state_dict": model_fp16.state_dict(),
        "config": config,
        "precision": "fp16",
    }, output_dir / "checkpoint.pt")

    compression_ratio = fp32_size / fp16_size if fp16_size > 0 else 1.0

    # Save metadata
    metadata = {
        "compression_method": "fp16",
        "precision": "float16",
        "original_model_type": checkpoint["model_type"],
        "fp32_size_mb": round(fp32_size, 2),
        "fp16_size_mb": round(fp16_size, 2),
        "compression_ratio": round(compression_ratio, 2),
        "device_support": "cuda",
        "original_parameters": model.get_num_parameters(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved to: {output_dir}")
    print(f"  Original size (FP32): {fp32_size:.2f} MB")
    print(f"  FP16 size: {fp16_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"\nTo run inference:")
    print(f"  model = model.half().cuda()")
    print(f"  inputs = inputs.cuda()")
    print(f"  with torch.autocast('cuda'):")
    print(f"      outputs = model(inputs)")


if __name__ == "__main__":
    main()