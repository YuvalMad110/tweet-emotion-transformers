"""
trainer.py - Training and evaluation for emotion classification.
Handles training loop, validation, early stopping, logging, checkpointing, and visualization.
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

from models import EmotionClassifier, ID2LABEL, NUM_LABELS
from utils.utils import get_israel_timestamp, get_project_root


class EarlyStopping:
    """Early stopping handler based on validation metric."""
    
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' for metrics like F1 (higher is better), 'min' for loss
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if score improved. Returns True if this is the best model so far."""
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class Trainer:
    """
    Trainer class for emotion classification models.
    
    Handles:
        - Training and validation loops with unified run_epoch
        - Early stopping based on macro F1
        - Checkpointing best model
        - Logging to file and console
        - Tracking and plotting metrics
        - Final evaluation with classification report
    """
    
    def __init__(self, model: EmotionClassifier, train_loader: DataLoader, val_loader: DataLoader,
                 output_dir: str, learning_rate: float = 2e-5, epochs: int = 10,
                 weight_decay: float = 0.01, warmup_ratio: float = 0.1, max_grad_norm: float = 1.0,
                 patience: int = 3, class_weights: Optional[torch.Tensor] = None,
                 device: str = "cuda", experiment_name: Optional[str] = None,
                 config_args: object = None):
        """
        Args:
            model: EmotionClassifier instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            output_dir: Directory to save model, logs, and plots
            learning_rate: Learning rate for AdamW optimizer
            epochs: Maximum number of training epochs
            weight_decay: Weight decay for regularization
            warmup_ratio: Ratio of total steps for learning rate warmup
            max_grad_norm: Maximum gradient norm for clipping
            patience: Early stopping patience (epochs without improvement)
            class_weights: Optional tensor of class weights for imbalanced data
            device: Device to train on ('cuda' or 'cpu')
            experiment_name: Name for this experiment (used in file names)
            config_args: Optional argparse.Namespace with all run arguments (for full reproducibility)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Setup output directory
        self.experiment_name = experiment_name or f"{model.model_type.name}_{get_israel_timestamp()}"
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function with optional class weights
        weight_tensor = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        
        # Tracking history
        self.history = {
            "train_loss": [], "train_accuracy": [], "train_macro_f1": [],
            "val_loss": [], "val_accuracy": [], "val_macro_f1": [], 
            "val_weighted_f1": [], "val_avg_recall": [], "learning_rate": []
        }
        
        # Store config for logging - full args for exact reproducibility
        self.config = vars(config_args).copy()
        self.config["num_parameters"] = model.get_num_parameters()
        self.config["trainable_parameters"] = model.get_num_parameters(trainable_only=True)
        self.config["train_samples"] = len(train_loader.dataset)
        self.config["val_samples"] = len(val_loader.dataset)
        
        # Setup logging
        self._setup_logging()
        self._log_config()
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "training.log"
        
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _log_config(self):
        """Log experiment configuration."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("=" * 60)
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60 + "\n")
        
        # Save config as JSON
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _run_epoch(self, data_loader: DataLoader, training: bool = True) -> Dict[str, float]:
        """
        Run a single epoch (train or eval).
        
        Args:
            data_loader: DataLoader to iterate over
            training: If True, run training (backprop + optimizer step). If False, evaluation only.
            
        Returns:
            Dictionary with loss, accuracy, macro_f1, weighted_f1, avg_recall
        """
        if training:
            self.model.train()
            desc = "Training"
        else:
            self.model.eval()
            desc = "Validating"
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        context = torch.no_grad() if not training else torch.enable_grad()
        
        with context:
            progress_bar = tqdm(data_loader, desc=desc, leave=False)
            for input_ids, attention_mask, labels in progress_bar:
                # Move to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs["logits"], labels)
                
                if training:
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                preds = outputs["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro"),
            "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
            "avg_recall": recall_score(all_labels, all_preds, average="macro"),
        }
        
        return metrics
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...\n")
        
        best_model_state = None
        best_epoch = 0
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Training
            train_metrics = self._run_epoch(self.train_loader, training=True)
            
            # Validation
            val_metrics = self._run_epoch(self.val_loader, training=False)
            
            # Track history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["train_macro_f1"].append(train_metrics["macro_f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            self.history["val_avg_recall"].append(val_metrics["avg_recall"])
            self.history["learning_rate"].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
            self.logger.info(f"  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['macro_f1']:.4f}")
            self.logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['macro_f1']:.4f} | AvgRec: {val_metrics['avg_recall']:.4f}")
            
            # Early stopping check
            is_best = self.early_stopping(val_metrics["macro_f1"])
            if is_best:
                self.logger.info(f"  ✓ New best model (Macro-F1: {val_metrics['macro_f1']:.4f})")
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
            
            self.logger.info("")
            
            if self.early_stopping.should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        
        # Restore best model and save checkpoint
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.best_model_state = best_model_state
            self.best_epoch = best_epoch
        
        # Save checkpoint (safety net in case evaluation fails)
        self._save_checkpoint()
        
        # Save history and plots
        self._save_history()
        self._plot_metrics()
        
        # Final summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        self.logger.info(f"  Best epoch: {best_epoch}")
        self.logger.info(f"  Best val Macro-F1: {self.early_stopping.best_score:.4f}")
        self.logger.info("=" * 60)
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader, save_predictions: bool = True) -> Dict:
        """
        Full evaluation with classification report and confusion matrix.
        
        Args:
            test_loader: DataLoader for test data
            save_predictions: If True, save predictions to file
            
        Returns:
            Dictionary with all metrics, predictions, and confusion matrix
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 60)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_loader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = outputs["logits"].argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Compute metrics
        label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
        report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro"),
            "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
            "avg_recall": recall_score(all_labels, all_preds, average="macro"),
            "predictions": all_preds,
            "labels": all_labels,
            "confusion_matrix": conf_matrix,
        }
        
        # Log results
        self.logger.info(f"\nAccuracy: {results['accuracy']:.4f}")
        self.logger.info(f"Macro F1: {results['macro_f1']:.4f}")
        self.logger.info(f"Weighted F1: {results['weighted_f1']:.4f}")
        self.logger.info(f"Avg Recall: {results['avg_recall']:.4f}")
        self.logger.info(f"\nClassification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Save results
        if save_predictions:
            self._save_evaluation_results(results, report)
            self._plot_confusion_matrix(conf_matrix, label_names)
        
        return results
    
    def _save_history(self):
        """Save training history to JSON file."""
        history_file = self.output_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_file}")
    
    def _save_checkpoint(self, eval_results: Dict = None):
        """
        Save comprehensive checkpoint with all info needed for inference and reproducibility.
        
        Checkpoint contains:
            - state_dict: Model weights
            - config: All run configuration (for exact reproduction)
            - model_type: Model type string (for loading correct architecture)
            - best_epoch: Epoch of best model
            - history: Training history
            - eval_results: Evaluation metrics (if available)
        """
        state_dict = getattr(self, 'best_model_state', None) or self.model.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        checkpoint = {
            "state_dict": state_dict,
            "config": self.config,
            "model_type": self.model.model_type.name,
            "best_epoch": getattr(self, 'best_epoch', None),
            "history": self.history,
            "eval_results": eval_results,
        }
        
        checkpoint_path = self.output_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_evaluation_results(self, results: Dict, report: str):
        """Save evaluation results and final checkpoint."""
        # Prepare metrics
        metrics_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels", "confusion_matrix"]}
        metrics_to_save["confusion_matrix"] = results["confusion_matrix"].tolist()
        metrics_to_save["classification_report"] = report

        # Save metrics JSON
        metrics_file = self.output_dir / "eval_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        # Save predictions JSON - convert numpy types to native Python types
        preds_file = self.output_dir / "predictions.json"
        with open(preds_file, 'w') as f:
            json.dump({
                "predictions": [int(p) for p in results["predictions"]],
                "labels": [int(l) for l in results["labels"]]
            }, f)
        
        # Save final checkpoint with everything
        self._save_checkpoint(eval_results=metrics_to_save)
        
        self.logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def _plot_metrics(self):
        """Plot and save training/validation metrics."""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(epochs, self.history["train_loss"], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history["val_loss"], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history["train_accuracy"], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history["val_accuracy"], 'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Macro F1
        axes[1, 0].plot(epochs, self.history["train_macro_f1"], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history["val_macro_f1"], 'r-', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Macro F1')
        axes[1, 0].set_title('Macro F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history["learning_rate"], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        self.logger.info(f"Training curves saved to {plot_file}")
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray, label_names: List[str]):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(conf_matrix.shape[1]),
               yticks=np.arange(conf_matrix.shape[0]),
               xticklabels=label_names, yticklabels=label_names,
               xlabel='Predicted', ylabel='True',
               title='Confusion Matrix')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plot_file = self.output_dir / "confusion_matrix.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {plot_file}")