"""
run_inference.py - Inference script for emotion classification.
Loads trained DeBERTa-Large weights and runs predictions on a CSV file.

Usage:
    python run_inference.py --weights <path_to_weights> --csv <path_to_csv>

    Or use as a module:
        from run_inference import run_inference
        predictions = run_inference(weights_path, csv_path)
"""

import argparse
import csv
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

from models import EmotionClassifier, ModelType, ID2LABEL


def run_inference(
    weights: str,
    csv_path: str,
    output_path: str = "predictions.csv",
    batch_size: int = 1,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Run inference on a CSV file using trained DeBERTa-Large model weights.

    Args:
        weights: Path to model weights file (checkpoint.pt)
        csv_path: Path to CSV file with 'text' column
        output_path: Path to save predictions CSV
        batch_size: Batch size for inference
        device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        List of dictionaries with 'text', 'predicted_label', and 'predicted_emotion' for each sample
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer (hardcoded to DeBERTa-Large)
    model_type = ModelType.DEBERTA_LARGE
    print(f"Loading model: {model_type.value}")

    # Load checkpoint - support both nested format (with "state_dict" key) and direct state_dict
    checkpoint = torch.load(weights, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = EmotionClassifier(model_type=model_type)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_type.value)
    print(f"Model loaded: {model.get_num_parameters():,} parameters")

    # Load CSV data
    texts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
    print(f"Loaded {len(texts)} samples from {csv_path}")

    # Run inference in batches
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Get predictions
            predicted_labels = torch.argmax(logits, dim=1).cpu().tolist()

            # Store results
            for text, label in zip(batch_texts, predicted_labels):
                predictions.append({
                    "text": text,
                    "predicted_label": label,
                    "predicted_emotion": ID2LABEL[label]
                })

    print(f"Inference complete: {len(predictions)} predictions")

    # Save predictions
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "predicted_label", "predicted_emotion"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to: {output_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained DeBERTa-Large model")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to model weights (checkpoint.pt)")
    parser.add_argument("--csv", "-c", type=str, required=True, help="Path to input CSV file with 'text' column")
    parser.add_argument("--output_path", "-o", type=str, default="predictions.csv", help="Path to output predictions CSV")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["cuda", "cpu"], help="Device to run on")
    args = parser.parse_args()

    predictions = run_inference(
        weights=args.weights,
        csv_path=args.csv,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device
    )

    # Print summary
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)

    from collections import Counter
    label_counts = Counter(p["predicted_emotion"] for p in predictions)

    print(f"Total predictions: {len(predictions)}")
    print("\nDistribution:")
    for emotion in ID2LABEL.values():
        count = label_counts.get(emotion, 0)
        pct = count / len(predictions) * 100
        print(f"  {emotion}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
