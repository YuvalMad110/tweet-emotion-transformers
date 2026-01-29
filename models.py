"""
models.py - Transformer models for emotion classification.
Supports DeBERTa, BERTweet, ELECTRA in base/small/large variants.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Dict, Tuple
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    # DeBERTa variants
    DEBERTA_SMALL = "microsoft/deberta-small" # "microsoft/deberta-v3-small"
    DEBERTA_BASE = "microsoft/deberta-base"
    DEBERTA_LARGE = "microsoft/deberta-large"
    # BERTweet (only base available)
    BERTWEET_BASE = "vinai/bertweet-base"
    BERTWEET_LARGE = "vinai/bertweet-large"
    # ELECTRA variants
    ELECTRA_SMALL = "google/electra-small-discriminator"
    ELECTRA_BASE = "google/electra-base-discriminator"
    ELECTRA_LARGE = "google/electra-large-discriminator"


# Label mappings
LABEL2ID = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


class EmotionClassifier(nn.Module):
    """
    Unified emotion classifier for transformer models.
    Same classification head (Dropout -> Linear) for all models to ensure fair comparison.
    """
    
    def __init__(self, model_type: ModelType, num_labels: int = NUM_LABELS,
                 dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.model_type = model_type
        self.model_name = model_type.value
        self.num_labels = num_labels
        
        # Load pretrained encoder
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass. input_ids/attention_mask: (batch, seq_len), labels: (batch,). Returns dict with 'logits' and optionally 'loss'."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(self.dropout(pooled_output))
        
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def save(self, path: str):
        """Save model weights to path."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def from_pretrained(cls, path: str, model_type: ModelType, **kwargs) -> "EmotionClassifier":
        """Load a fine-tuned model from checkpoint."""
        model = cls(model_type=model_type, **kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model


def create_model_and_tokenizer(model_type: ModelType, dropout: float = 0.1,
                                freeze_encoder: bool = False) -> Tuple[EmotionClassifier, AutoTokenizer]:
    """Create model and tokenizer pair according to given model_type."""
    model = EmotionClassifier(model_type=model_type, dropout=dropout, freeze_encoder=freeze_encoder)
    tokenizer = AutoTokenizer.from_pretrained(model_type.value)
    return model, tokenizer


def get_model_info(model_type: ModelType) -> Dict:
    """Get model info without loading weights. Useful for checking specs before training."""
    config = AutoConfig.from_pretrained(model_type.value)
    return {
        "name": model_type.name,
        "pretrained_name": model_type.value,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
    }


if __name__ == "__main__":
    print("Testing model creation...\n")
    
    # Test only base models
    base_models = [ModelType.DEBERTA_BASE, ModelType.BERTWEET_BASE, ModelType.ELECTRA_BASE]
    
    for model_type in base_models:
        print(f"{'='*50}\nModel: {model_type.name}\n{'='*50}")
        
        info = get_model_info(model_type)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        model, tokenizer = create_model_and_tokenizer(model_type)
        print(f"  total_params: {model.get_num_parameters():,}")
        print(f"  trainable_params: {model.get_num_parameters(trainable_only=True):,}")
        
        # Test forward pass
        sample_text = "I am so happy today! 😊"
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        print(f"  sample: '{sample_text}'")
        print(f"  predicted: {ID2LABEL[outputs['logits'].argmax().item()]}\n")