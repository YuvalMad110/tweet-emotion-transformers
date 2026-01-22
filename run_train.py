"""
run_train.py - Main training script for emotion classification.
All configurations are passed via command line arguments.

Example usage:
    python train.py --model_type DEBERTA_BASE --epochs 10 --batch_size 32 --use_class_weights
    python train.py --model_type BERTWEET_BASE --learning_rate 3e-5 --output_dir outputs
    python train.py --model_type ELECTRA_BASE --eval_only --load_path checkpoints/electra_best.pt
"""

import argparse
import random
import numpy as np
import torch
import os

from models import ModelType, create_model_and_tokenizer
from data.dataset import get_data_loaders
from trainer import Trainer
from utils.utils import get_project_root


def parse_args() -> argparse.Namespace:
    """Parse all command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train transformer models for emotion classification",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_type", type=str, default="DEBERTA_BASE", choices=[m.name for m in ModelType], help="Pretrained model to use")
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for classification head")
    model_group.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights (train only classification head)")
    model_group.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for tokenization")
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train_path", type=str, default="/home/yuvalmad/datasets/tweets/train.csv", help="Path to training CSV file")
    data_group.add_argument("--val_path", type=str, default="/home/yuvalmad/datasets/tweets/validation.csv", help="Path to validation CSV file")
    data_group.add_argument("--test_path", type=str, default=None, help="Path to test CSV file (optional)")
    data_group.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    data_group.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs")
    train_group.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW optimizer")
    train_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    train_group.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for LR warmup")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    train_group.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    train_group.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    train_group.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "constant"],
                            help="LR scheduler: 'linear' (warmup+decay), 'cosine' (warmup+cosine decay), 'constant' (no decay)")
    
    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", type=str, default="outputs", help="Directory to save model, logs, and plots")
    output_group.add_argument("--experiment_name", type=str, default=None, help="Name for this experiment (default: auto-generated)")
    
    # Load/Eval arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--load_path", type=str, default=None, help="Path to load pretrained model weights, if None train from scratch")
    eval_group.add_argument("--eval_only", action="store_true", help="Only run evaluation (requires --load_path)")
    
    # Other arguments
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    other_group.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.eval_only and args.load_path is None:
        parser.error("--eval_only requires --load_path")
    
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_config(args: argparse.Namespace):
    """Print configuration in organized format."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


def main():

    # Parse arguments
    args = parse_args()
    set_seed(args.seed)
    print_config(args)

    # Resolve output directory relative to project root
    project_root = get_project_root()
    output_dir = os.path.join(project_root, args.output_dir)

    # Setup model
    model_type = ModelType[args.model_type]
    print(f"Loading model: {model_type.value}")
    model, tokenizer = create_model_and_tokenizer(model_type, dropout=args.dropout, freeze_encoder=args.freeze_encoder)
    if args.load_path:
        print(f"Loading weights from: {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location="cpu"))
    print(f"Model parameters: {model.get_num_parameters():,} (trainable: {model.get_num_parameters(trainable_only=True):,})\n")
    
    # Setup data loaders and class weights
    print("Loading data...")
    data = get_data_loaders(
        train_path=args.train_path, val_path=args.val_path, test_path=args.test_path,
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers
    )
    
    class_weights = data["class_weights"] if args.use_class_weights else None
    
    # Setup trainer
    trainer = Trainer(
        model=model, train_loader=data["train_loader"], val_loader=data["val_loader"],
        output_dir=output_dir, learning_rate=args.learning_rate, epochs=args.epochs,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, max_grad_norm=args.max_grad_norm,
        patience=args.patience, class_weights=class_weights, device=args.device,
        experiment_name=args.experiment_name, config_args=args, scheduler_type=args.scheduler
    )
    
    # Train
    if not args.eval_only:
        trainer.train()
    
    # Evaluate
    test_loader = data["test_loader"] if data["test_loader"] is not None else data["val_loader"]
    eval_set_name = "test" if data["test_loader"] is not None else "validation"
    print(f"\nEvaluating on {eval_set_name} set...")
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()