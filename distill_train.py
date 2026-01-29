"""
distill_train.py - Knowledge distillation training.
Train a smaller student model using a larger teacher model's soft predictions.
Saves to outputs/{experiment}/compressions/distilled_{student_type}/

Example usage:
    python distill_train.py --teacher DEBERTA_LARGE_09-01-54 --student_type DEBERTA_SMALL
    python distill_train.py --teacher DEBERTA_LARGE_09-01-54 --student_type DEBERTA_BASE --temperature 4.0
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from models import ModelType, EmotionClassifier
from data.dataset import get_data_loaders
from utils.utils import get_project_root


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    Loss = alpha * hard_loss + (1-alpha) * soft_loss
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (cross-entropy with true labels)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft loss (KL divergence with temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def get_model_size_mb(model) -> float:
    """Get model size in MB."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def train_distillation(teacher, student, train_loader, val_loader, args, device):
    """Train student with knowledge distillation."""

    teacher.to(device)
    student.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    print("Starting distillation training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Training
        student.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids, attention_mask)["logits"]

            student_logits = student(input_ids, attention_mask)["logits"]
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        student.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                teacher_logits = teacher(input_ids, attention_mask)["logits"]
                student_logits = student(input_ids, attention_mask)["logits"]

                val_loss += criterion(student_logits, teacher_logits, labels).item()
                all_preds.extend(student_logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping check
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    # Restore best model
    if best_state:
        student.load_state_dict(best_state)

    print(f"Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "training_time": training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation training")
    parser.add_argument("--teacher", type=str, default='DEBERTA_LARGE_09-01-54', help="Teacher experiment folder inside outputs/")
    parser.add_argument("--student_type", type=str, default="DEBERTA_BASE", choices=[m.name for m in ModelType])
    parser.add_argument("--temperature", type=float, default=3.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hard loss (1-alpha for soft loss)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_path", type=str, default="/home/yuvalmad/datasets/tweets/train.csv")
    parser.add_argument("--val_path", type=str, default="/home/yuvalmad/datasets/tweets/validation.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(get_project_root())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load teacher
    print("Loading teacher model...")
    teacher_dir = project_root / "outputs" / args.teacher
    teacher_checkpoint = torch.load(teacher_dir / "checkpoint.pt", map_location="cpu")
    teacher_type = ModelType[teacher_checkpoint["model_type"]]
    teacher = EmotionClassifier(model_type=teacher_type)
    teacher.load_state_dict(teacher_checkpoint["state_dict"])

    # Create student
    print("Creating student model...")
    student_type = ModelType[args.student_type]
    student = EmotionClassifier(model_type=student_type)

    # Load data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(student_type.value)
    data = get_data_loaders(
        train_path=args.train_path, val_path=args.val_path,
        tokenizer=tokenizer, batch_size=args.batch_size,
    )

    # Get sizes before training
    teacher_size = get_model_size_mb(teacher)
    student_size = get_model_size_mb(student)
    teacher_params = teacher.get_num_parameters()
    student_params = student.get_num_parameters()

    # Train
    results = train_distillation(
        teacher, student, data["train_loader"], data["val_loader"], args, device
    )

    # Create output directory
    output_dir = teacher_dir / "compressions" / f"distilled_{args.student_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save student checkpoint
    print("Saving student model...")
    torch.save({
        "state_dict": student.state_dict(),
        "model_type": args.student_type,
    }, output_dir / "checkpoint.pt")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(results["history"], f, indent=2)

    # Save metadata
    metadata = {
        "compression_method": "knowledge_distillation",
        "teacher_experiment": args.teacher,
        "teacher_model_type": teacher_checkpoint["model_type"],
        "student_model_type": args.student_type,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "epochs_trained": results["best_epoch"],
        "best_val_f1": round(results["best_f1"], 4),
        "training_time_seconds": round(results["training_time"], 1),
        "teacher_size_mb": round(teacher_size, 2),
        "student_size_mb": round(student_size, 2),
        "compression_ratio": round(teacher_size / student_size, 2),
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Saved to: {output_dir}")
    print(f"  Teacher: {teacher_size:.2f} MB ({teacher_params:,} params)")
    print(f"  Student: {student_size:.2f} MB ({student_params:,} params) → {teacher_size/student_size:.2f}x compression")


if __name__ == "__main__":
    main()