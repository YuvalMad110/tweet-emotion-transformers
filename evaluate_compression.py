"""
evaluate_compression.py - Compare original model with compressed versions.
Supports ONNX quantization, FP16 quantization, and knowledge distillation comparisons.

Example usage:
    # Evaluate ONNX dynamic quantization only (CPU)
        python evaluate_compression.py --experiment DEBERTA_LARGE_09-01-54 --use-quantized true --quant_type dynamic --device cpu
    # Evaluate FP16 quantization only (GPU)
        python evaluate_compression.py --experiment DEBERTA_LARGE_09-01-54 --use-fp16 true --device cuda
    # Evaluate distillation only
        python evaluate_compression.py --experiment DEBERTA_LARGE_09-01-54 --use-distilled true --student_type DEBERTA_BASE
    # Compare FP16 and distillation on GPU
        python evaluate_compression.py --experiment DEBERTA_LARGE_09-01-54 --use-fp16 true --use-distilled true --student_type DEBERTA_BASE --device cuda
"""

import argparse
import json
import time
import torch
import numpy as np
# import onnxruntime as ort
from pathlib import Path
from datetime import datetime

from models import ModelType, EmotionClassifier, ID2LABEL, NUM_LABELS
from data.dataset import get_data_loaders
from utils.utils import get_project_root, str2bool
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


def get_model_size_mb(model) -> float:
    """Get model size in MB."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def measure_inference_time(model, data_loader, device, num_batches=20):
    """Measure average inference time per batch."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (input_ids, attention_mask, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms per batch


def evaluate_model(model, data_loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "report": classification_report(all_labels, all_preds,
                                        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=4),
    }


def load_original_model(experiment_dir):
    """Load original model from experiment directory."""
    checkpoint = torch.load(experiment_dir / "checkpoint.pt", map_location="cpu")
    model_type = ModelType[checkpoint["model_type"]]
    config = checkpoint.get("config", {})

    model = EmotionClassifier(model_type=model_type, dropout=config.get("dropout", 0.1))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, model_type, checkpoint


def load_onnx_model(experiment_dir, quant_type="dynamic", device="cuda"):
    """Load ONNX quantized model from compressions folder."""
    onnx_dir = experiment_dir / "compressions" / f"onnx_{quant_type}"
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_dir}")

    model_path = onnx_dir / "model.onnx"
    metadata = json.load(open(onnx_dir / "metadata.json"))

    # Create ONNX Runtime session with TensorRT for GPU-accelerated INT8 inference
    if device == "cuda":
        # TensorRT engine cache directory (avoids rebuilding on subsequent runs)
        trt_cache_dir = str(onnx_dir / "trt_cache")
        Path(trt_cache_dir).mkdir(exist_ok=True)

        providers = [
            ("TensorrtExecutionProvider", {
                "trt_int8_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)

    # Report actual provider being used
    actual_providers = session.get_providers()
    metadata["onnx_providers"] = actual_providers
    print(f"  ONNX Runtime providers: {actual_providers}")

    return session, metadata, model_path


def get_onnx_model_size_mb(model_path: Path) -> float:
    """Get ONNX model file size in MB."""
    return model_path.stat().st_size / (1024 * 1024)


def measure_onnx_inference_time(session, data_loader, num_batches=20):
    """Measure average inference time per batch for ONNX model."""
    times = []
    warmup_done = False

    for i, (input_ids, attention_mask, _) in enumerate(data_loader):
        if i >= num_batches + 1:  # +1 for warmup
            break

        ort_inputs = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        }

        if not warmup_done:
            # Warmup run to trigger TensorRT engine building (not timed)
            _ = session.run(None, ort_inputs)
            warmup_done = True
            continue

        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms per batch


def evaluate_onnx_model(session, data_loader):
    """Evaluate ONNX model and return metrics."""
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in tqdm(data_loader, desc="Evaluating ONNX", leave=False):
        ort_inputs = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        }
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "report": classification_report(all_labels, all_preds,
                                        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=4),
    }


def load_distilled_model(experiment_dir, student_type):
    """Load distilled model from compressions folder."""
    distill_dir = experiment_dir / "compressions" / f"distilled_{student_type}"
    if not distill_dir.exists():
        raise FileNotFoundError(f"Distilled model not found at {distill_dir}")

    checkpoint = torch.load(distill_dir / "checkpoint.pt", map_location="cpu")
    metadata = json.load(open(distill_dir / "metadata.json"))

    model_type = ModelType[checkpoint["model_type"]]
    model = EmotionClassifier(model_type=model_type)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, metadata


def load_fp16_model(experiment_dir):
    """Load FP16 quantized model from compressions folder."""
    fp16_dir = experiment_dir / "compressions" / "fp16"
    if not fp16_dir.exists():
        raise FileNotFoundError(f"FP16 model not found at {fp16_dir}")

    checkpoint = torch.load(fp16_dir / "checkpoint.pt", map_location="cpu")
    metadata = json.load(open(fp16_dir / "metadata.json"))

    model_type = ModelType[checkpoint["model_type"]]
    config = checkpoint.get("config", {})
    model = EmotionClassifier(model_type=model_type, dropout=config.get("dropout", 0.1))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.half()  # Ensure FP16
    model.eval()

    return model, metadata


def generate_report(report_path, experiment_name, original_type, results, device,
                    use_quantized, use_distilled, student_type, quant_type="dynamic"):
    """Generate comparison report and save to file."""
    original_size = results["Original"]["size_mb"]
    original_time = results["Original"]["inference_ms"]
    original_f1 = results["Original"]["metrics"]["macro_f1"]

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPRESSION COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Original model: {original_type.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation device: {device}\n\n")

        # Compression methods info
        f.write("COMPRESSION METHODS\n")
        f.write("-" * 70 + "\n")

        if use_quantized and results["Quantized"]["metadata"]:
            meta = results["Quantized"]["metadata"]
            if quant_type == "fp16":
                f.write(f"FP16 Quantization:\n")
                f.write(f"  Method: {meta.get('compression_method', 'fp16')}\n")
                f.write(f"  Precision: {meta.get('precision', 'float16')}\n")
                f.write(f"  FP32 size: {meta.get('fp32_size_mb', 'N/A')} MB\n")
                f.write(f"  FP16 size: {meta.get('fp16_size_mb', 'N/A')} MB\n")
                f.write(f"  Device support: {meta.get('device_support', 'cuda')}\n")
            else:
                f.write(f"ONNX Quantization:\n")
                f.write(f"  Method: {meta.get('compression_method', f'onnx_{quant_type}')}\n")
                f.write(f"  Quantization type: {meta.get('quantization_type', quant_type)}\n")
                f.write(f"  FP32 size: {meta.get('fp32_size_mb', 'N/A')} MB\n")
                f.write(f"  Quantized size: {meta.get('quantized_size_mb', 'N/A')} MB\n")
                if "onnx_providers" in meta:
                    f.write(f"  Execution provider: {meta['onnx_providers'][0]}\n")
            f.write("\n")

        if use_distilled and results["Distilled"]["metadata"]:
            meta = results["Distilled"]["metadata"]
            f.write(f"Knowledge Distillation:\n")
            f.write(f"  Student model: {meta.get('student_model_type', student_type)}\n")
            f.write(f"  Temperature: {meta.get('temperature', 'N/A')}\n")
            f.write(f"  Alpha: {meta.get('alpha', 'N/A')}\n")
            f.write(f"  Epochs trained: {meta.get('epochs_trained', 'N/A')}\n")
            f.write(f"  Training time: {meta.get('training_time_seconds', 'N/A')}s\n\n")

        # Size comparison
        f.write("MODEL SIZE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<15} {'Size (MB)':>12} {'Parameters':>15} {'Compression':>12}\n")
        f.write("-" * 70 + "\n")

        for name, res in results.items():
            compression = f"{original_size / res['size_mb']:.2f}x" if name != "Original" else "-"
            params_str = f"{res['parameters']:,}" if res['parameters'] else "-"
            f.write(f"{name:<15} {res['size_mb']:>12.2f} {params_str:>15} {compression:>12}\n")
        f.write("\n")

        # Inference time comparison
        f.write(f"INFERENCE TIME (per batch, {device})\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<15} {'Time (ms)':>12} {'Speedup':>12}\n")
        f.write("-" * 70 + "\n")

        for name, res in results.items():
            speedup = f"{original_time / res['inference_ms']:.2f}x" if name != "Original" else "-"
            f.write(f"{name:<15} {res['inference_ms']:>12.2f} {speedup:>12}\n")
        f.write("\n")

        # Accuracy comparison
        f.write("EVALUATION METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<15} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'F1 Change':>12}\n")
        f.write("-" * 70 + "\n")

        for name, res in results.items():
            m = res["metrics"]
            f1_change = f"{m['macro_f1'] - original_f1:+.4f}" if name != "Original" else "-"
            f.write(f"{name:<15} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f} {f1_change:>12}\n")
        f.write("\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")

        for name, res in results.items():
            if name == "Original":
                continue

            compression = original_size / res['size_mb']
            speedup = original_time / res['inference_ms']
            f1_change = res['metrics']['macro_f1'] - original_f1

            f.write(f"{name}:\n")
            f.write(f"  Size reduction: {compression:.2f}x ({original_size:.1f} MB -> {res['size_mb']:.1f} MB)\n")
            f.write(f"  Speedup: {speedup:.2f}x ({original_time:.1f} ms -> {res['inference_ms']:.1f} ms)\n")
            f.write(f"  F1 change: {f1_change:+.4f} ({original_f1:.4f} -> {res['metrics']['macro_f1']:.4f})\n\n")

        # Per-class breakdown for each model
        f.write("PER-CLASS CLASSIFICATION REPORTS\n")
        f.write("-" * 70 + "\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(res["metrics"]["report"])
            f.write("\n")

        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare compressed models")
    parser.add_argument("--experiment", type=str, default='DEBERTA_LARGE_09-01-54', help="Experiment folder inside outputs/")
    parser.add_argument("--use-quantized", type=str2bool, default=True, help="Include ONNX quantized model (true/false)")
    parser.add_argument("--quant_type", type=str, default="fp16", choices=["dynamic", "static", "fp32", "fp16"],
                        help="Quantization type: dynamic/static/fp32 (ONNX, CPU), fp16 (PyTorch, GPU)")
    parser.add_argument("--use-distilled", type=str2bool, default=True, help="Include distilled model (true/false)")
    parser.add_argument("--student_type", type=str, default="DEBERTA_BASE", help="Student model type for distillation")
    parser.add_argument("--val_path", type=str, default="/home/yuvalmad/datasets/tweets/validation.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.use_distilled and not args.use_quantized:
        parser.error("At least one of --use-distilled true or --use-quantized true must be specified")

    if args.use_distilled and not args.student_type:
        parser.error("--student_type is required when --use-distilled true")

    project_root = Path(get_project_root())
    experiment_dir = project_root / "outputs" / args.experiment
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Load original model
    print("Loading original model...")
    original_model, original_type, _ = load_original_model(experiment_dir)
    original_params = original_model.get_num_parameters()
    original_model.to(device)

    # Load data
    print("Loading validation data...")
    tokenizer = AutoTokenizer.from_pretrained(original_type.value)
    data = get_data_loaders(
        train_path=args.val_path, val_path=args.val_path,
        tokenizer=tokenizer, batch_size=args.batch_size,
    )
    val_loader = data["val_loader"]

    # Evaluate original model first
    print("Evaluating Original model...")
    results = {
        "Original": {
            "size_mb": get_model_size_mb(original_model),
            "parameters": original_params,
            "inference_ms": measure_inference_time(original_model, val_loader, device),
            "metrics": evaluate_model(original_model, val_loader, device),
            "metadata": None,
        }
    }

    # Evaluate quantized model
    if args.use_quantized:
        if args.quant_type == "fp16":
            # FP16 is a PyTorch model, not ONNX
            print("Loading FP16 quantized model...")
            fp16_model, fp16_metadata = load_fp16_model(experiment_dir)
            fp16_model.to(device)

            print("Evaluating Quantized (FP16) model...")
            results["Quantized"] = {
                "size_mb": fp16_metadata.get("fp16_size_mb", get_model_size_mb(fp16_model)),
                "parameters": fp16_metadata.get("original_parameters", original_params),
                "inference_ms": measure_inference_time(fp16_model, val_loader, device),
                "metrics": evaluate_model(fp16_model, val_loader, device),
                "metadata": fp16_metadata,
            }
        else:
            # ONNX quantization (dynamic, static, fp32)
            print(f"Loading ONNX {args.quant_type} quantized model...")
            onnx_session, onnx_metadata, onnx_path = load_onnx_model(experiment_dir, args.quant_type, device_str)

            print("Evaluating Quantized (ONNX) model...")
            results["Quantized"] = {
                "size_mb": get_onnx_model_size_mb(onnx_path),
                "parameters": onnx_metadata.get("original_parameters", original_params),
                "inference_ms": measure_onnx_inference_time(onnx_session, val_loader),
                "metrics": evaluate_onnx_model(onnx_session, val_loader),
                "metadata": onnx_metadata,
            }

    # Evaluate distilled model
    if args.use_distilled:
        print(f"Loading distilled model ({args.student_type})...")
        distill_model, distill_metadata = load_distilled_model(experiment_dir, args.student_type)
        distill_model.to(device)

        print("Evaluating Distilled model...")
        results["Distilled"] = {
            "size_mb": get_model_size_mb(distill_model),
            "parameters": distill_model.get_num_parameters(),
            "inference_ms": measure_inference_time(distill_model, val_loader, device),
            "metrics": evaluate_model(distill_model, val_loader, device),
            "metadata": distill_metadata,
        }

    # Generate report
    print("Generating report...")
    report_path = experiment_dir / "compressions" / "comparison_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(
        report_path=report_path,
        experiment_name=args.experiment,
        original_type=original_type,
        results=results,
        device=device,
        use_quantized=args.use_quantized,
        use_distilled=args.use_distilled,
        student_type=args.student_type,
        quant_type=args.quant_type,
    )

    print(f"Done. Report saved to: {report_path}")


if __name__ == "__main__":
    main()