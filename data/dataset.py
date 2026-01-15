"""
dataset.py - Dataset handling for transformer-based emotion classification.
Uses each model's pretrained tokenizer instead of custom vocabulary.
CSV format expected: columns 'text' and 'label'
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizer

from models import LABEL2ID, ID2LABEL, NUM_LABELS


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification with transformer tokenizers."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer,
                 max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns tokenized sample.
        
        Returns:
            input_ids: Token IDs, shape (max_length,)
            attention_mask: Attention mask, shape (max_length,)
            label: Class label, scalar tensor
        """
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (input_ids, attention_mask, label) tuples
        
    Returns:
        input_ids: (batch_size, max_length)
        attention_mask: (batch_size, max_length)
        labels: (batch_size,)
    """
    input_ids, attention_masks, labels = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(labels)
    )


def load_csv(file_path: str, verbose: bool = True) -> Tuple[List[str], List[int]]:
    """
    Load texts and labels from CSV file.
    
    Args:
        file_path: Path to CSV file with 'text' and 'label' columns
        verbose: If True, print distribution statistics
        
    Returns:
        texts: List of text strings
        labels: List of integer labels (0-5)
    """
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))
    
    if verbose:
        print(f"Loaded {len(texts)} samples from {file_path}")
        label_counts = Counter(labels)
        print("Label distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / len(labels) * 100
            print(f"  {label} ({ID2LABEL[label]}): {count} ({pct:.1f}%)")
    
    return texts, labels


def compute_class_weights(labels: List[int], num_classes: int = NUM_LABELS) -> torch.Tensor:
    """
    Compute class weights inversely proportional to frequency.
    
    Formula: w_c = N / (C × n_c)
    Where N = total samples, C = num classes, n_c = samples in class c.
    
    This gives higher weights to minority classes, so misclassifying a rare
    emotion (e.g., surprise at 3.6%) costs more than a common one (e.g., joy at 33.5%).
    
    Args:
        labels: List of all training labels
        num_classes: Number of classes (default 6)
        
    Returns:
        weights: Tensor of shape (num_classes,) with weight per class
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * counts.get(c, 1))
    return weights


def get_data_loaders(train_path: str, val_path: str, tokenizer: PreTrainedTokenizer,
                     batch_size: int = 32, max_length: Optional[int] = 128,
                     test_path: Optional[str] = None, num_workers: int = 0) -> Dict:
    """
    Main entry point for data preparation.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        tokenizer: Pretrained tokenizer from the model
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length for tokenization (default 128, uses model_max_length if None)
        test_path: Optional path to test CSV
        num_workers: Number of workers for DataLoader (0 for main process)

    Returns:
        Dictionary containing:
            - train_loader, val_loader, test_loader (DataLoaders)
            - class_weights: Tensor of shape (num_classes,)
            - train_size, val_size, test_size: Dataset sizes
    """
    # Ensure max_length has a valid value
    if max_length is None:
        max_length = min(getattr(tokenizer, 'model_max_length', 128), 512)
        print(f"Using max_length={max_length} (from tokenizer default)")

    # Load data
    train_texts, train_labels = load_csv(train_path)
    val_texts, val_labels = load_csv(val_path)
    
    test_texts, test_labels = None, None
    if test_path:
        test_texts, test_labels = load_csv(test_path)
    
    # Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length) if test_texts else None
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers)
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    
    print(f"\nDataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches", end="")
    if test_loader:
        print(f", test={len(test_loader)} batches")
    else:
        print()
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "num_classes": NUM_LABELS,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts) if test_texts else 0,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from models import ModelType
    
    print("Testing dataset module...\n")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ModelType.DEBERTA_BASE.value)
    
    print("Example usage:")
    print("  data = get_data_loaders('train.csv', 'val.csv', tokenizer, batch_size=32)")
    print("  for input_ids, attention_mask, labels in data['train_loader']:")
    print("      outputs = model(input_ids, attention_mask, labels)")