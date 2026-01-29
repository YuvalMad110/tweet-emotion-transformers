# Tweet Emotion Classification with Transformers

A PyTorch-based project for emotion classification in Twitter data using pre-trained transformer models (DeBERTa, BERTweet, ELECTRA), with support for model compression via FP16 quantization and knowledge distillation.

## Project Overview

This project classifies tweets into six emotion categories:
- **0**: sadness
- **1**: joy
- **2**: love
- **3**: anger
- **4**: fear
- **5**: surprise

**Best Model**: DeBERTa-Large achieves **94.45% accuracy** and **91.99% Macro F1-score**.

## Project Structure

```
tweet-emotion-transformers/
├── models.py              # Model definitions (EmotionClassifier, ModelType)
├── trainer.py             # Training loop, evaluation, logging
├── run_train.py           # Main training script (CLI)
├── run_inference.py       # Inference on new data
├── distill_train.py       # Knowledge distillation training
├── quantize_fp16.py       # FP16 quantization
├── quantize.py            # ONNX quantization (experimental)
├── evaluate_compression.py # Compare original vs compressed models
├── data/
│   └── dataset.py         # Dataset loading and preprocessing
├── utils/
│   └── utils.py           # Helper utilities
├── outputs/               # Experiment outputs (created during training)
│   └── <EXPERIMENT_NAME>/
│       ├── checkpoint.pt       # Model weights
│       ├── config.json         # Training configuration
│       ├── training.log        # Training logs
│       ├── history.json        # Loss/accuracy per epoch
│       ├── eval_metrics.json   # Final evaluation metrics
│       ├── predictions.json    # Per-sample predictions
│       ├── training_curves.png # Loss/accuracy plots
│       ├── confusion_matrix.png
│       └── compressions/       # Compressed model variants
└── report.md              # Project report
```

## Installation

```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

## Data Format

CSV files with two columns:
- `text`: Tweet text
- `label`: Emotion label (0-5)

```csv
text,label
i feel so happy today,1
this makes me sad,0
```

## Usage

### 1. Training

Train a transformer model on emotion classification:

```bash
# Train DeBERTa-Large (default, best performance)
python run_train.py --train_path data/train.csv --val_path data/val.csv

# Train BERTweet-Large
python run_train.py --model_type BERTWEET_LARGE --train_path data/train.csv --val_path data/val.csv

```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | DEBERTA_LARGE | Model: DEBERTA_SMALL/BASE/LARGE, BERTWEET_BASE/LARGE, ELECTRA_SMALL/BASE/LARGE |
| `--batch_size` | 8 | Training batch size |
| `--epochs` | 10 | Maximum training epochs |
| `--learning_rate` | 2e-5 | Learning rate |
| `--patience` | 4 | Early stopping patience |
| `--scheduler` | linear | LR scheduler: linear, cosine, constant |
| `--use_class_weights` | False | Use class weights for imbalanced data |

**Output:** Creates `outputs/<MODEL>_<TIMESTAMP>/` with model checkpoint and metrics.

### 2. Inference

Run predictions on new data using trained weights:

```bash
python run_inference.py --weights outputs/DEBERTA_LARGE_27-16-34/checkpoint.pt --csv test.csv
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--weights`, `-w` | required | Path to checkpoint.pt |
| `--csv`, `-c` | required | Input CSV with 'text' column |
| `--output_path`, `-o` | predictions.csv | Output predictions file |
| `--batch_size`, `-b` | 1 | Inference batch size |
| `--device`, `-d` | auto | cuda or cpu |

**Output:** CSV file with columns: `text`, `predicted_label`, `predicted_emotion`

**As Python module:**
```python
from run_inference import run_inference
predictions = run_inference("checkpoint.pt", "test.csv")
```

### 3. Model Compression

#### FP16 Quantization

Convert model to half-precision for faster GPU inference:

```bash
python quantize_fp16.py --experiment DEBERTA_LARGE_27-16-34
```

**Output:** Creates `outputs/<EXPERIMENT>/compressions/fp16/` with quantized model.

#### Knowledge Distillation

Train a smaller student model using the larger model as teacher:

```bash
python distill_train.py \
    --teacher DEBERTA_LARGE_27-16-34 \
    --student_type DEBERTA_BASE \
    --temperature 3.0 \
    --alpha 0.5 \
    --epochs 5
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher` | required | Teacher experiment folder name |
| `--student_type` | DEBERTA_BASE | Student model architecture |
| `--temperature` | 3.0 | Distillation temperature (higher = softer) |
| `--alpha` | 0.5 | Weight for hard labels (1-alpha for soft) |

**Output:** Creates `outputs/<TEACHER>/compressions/distilled_<STUDENT>/`

### 4. Compression Evaluation

Compare original, quantized, and distilled models:

```bash
# Evaluate both compression methods (after you ran distill_train and quantize_fp16)
python evaluate_compression.py \
    --experiment DEBERTA_LARGE_27-16-34 \
    --use-quantized true \
    --quant_type fp16 \
    --use-distilled true \
    --student_type DEBERTA_BASE
```

**Output:** Creates `outputs/<EXPERIMENT>/compressions/comparison_report.txt` with:
- Model sizes and compression ratios
- Inference latency and speedup
- Accuracy, F1-score comparison
- Per-class classification reports

## Output Files

After training, each experiment folder contains:

| File | Description |
|------|-------------|
| `checkpoint.pt` | Model weights (state_dict + metadata) |
| `config.json` | All training hyperparameters |
| `training.log` | Detailed training logs |
| `history.json` | Per-epoch metrics (loss, accuracy, F1) |
| `eval_metrics.json` | Final evaluation results |
| `predictions.json` | Per-sample predictions on validation set |
| `training_curves.png` | Loss and accuracy plots |
| `confusion_matrix.png` | Confusion matrix visualization |

## Results Summary

### Model Comparison

| Model | Parameters | Accuracy | Macro F1 |
|-------|------------|----------|----------|
| DeBERTa-Large | 405M | **94.25%** | **91.89%** |
| BERTweet-Large | 335M | 93.85% | 91.25% |

### Compression Results

| Method | Size | Compression | Speedup | F1 Change |
|--------|------|-------------|---------|-----------|
| Original | 1,546 MB | — | — | — |
| FP16 | 773 MB | 2.00× | 1.49× | +0.00% |
| Distilled | 529 MB | 2.92× | 3.42× | +0.19% |
