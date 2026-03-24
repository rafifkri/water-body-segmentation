# Semantic Segmentation - Satellite Water Body Detection

**Advanced PyTorch Implementation for Water Body Segmentation in Satellite Imagery**

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project provides a comprehensive, production-ready implementation of semantic segmentation for detecting and segmenting water bodies in satellite imagery using PyTorch. It includes multiple state-of-the-art architectures, advanced loss functions, data augmentation strategies, and a flexible configuration system.

### Key Capabilities
- **Multiple Architectures**: U-Net, SegNet with extensible design
- **Advanced Training**: Mixed precision, gradient accumulation, learning rate scheduling
- **Rich Configuration**: YAML/JSON config files with sensible defaults
- **Production Ready**: Comprehensive logging, checkpointing, and metrics tracking
- **Data Augmentation**: Albumentations-based augmentation pipeline
- **Flexible Evaluation**: Multiple metrics at various thresholds

---

## Features

### Model Architectures
- **U-Net**: Classic encoder-decoder with skip connections
- **Attention U-Net**: With SCSE/CBAM attention mechanisms
- **SegNet**: Efficient architecture with unpooling
- **DeepLabV3/V3+**: Coming soon
- **SegFormer**: Coming soon

### Loss Functions
- **Dice Loss**: For balanced segmentation
- **Focal Loss**: For handling class imbalance
- **BCE Loss**: Binary cross-entropy with weighting
- **Combined Loss**: Weighted combination of multiple losses
- **Tversky Loss**: Flexible alpha/beta weighting
- **Jaccard Loss**: IoU-based loss

### Training Features
- **Data Augmentation**: 10+ augmentation techniques (rotation, flip, elastic transform, etc.)
- **Learning Rate Scheduling**: Cosine, step, exponential, linear schedules
- **Early Stopping**: With configurable patience and delta
- **Mixed Precision Training**: Reduces memory usage and speeds up training
- **Gradient Clipping**: Prevents gradient explosion
- **Model Checkpointing**: Save best models and intermediate checkpoints

### Evaluation Metrics
- **IoU (Intersection over Union)**: Segmentation standard metric
- **Dice Coefficient**: F1-score equivalent for segmentation
- **Precision, Recall, F1-Score**: Classification metrics
- **Sensitivity, Specificity**: Medical imaging metrics
- **Matthews Correlation Coefficient**: Balanced classification metric
- **Per-image and Batch Metrics**: Detailed analysis

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Git

### Step 1: Clone or Setup Repository
```bash
cd "path/to/Satellite Water Body Semantic"
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n segmentation python=3.10
conda activate segmentation
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**For GPU acceleration (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Basic Training (Default Configuration)
```bash
python train.py
```

### Training with Custom Configuration
```bash
# Custom model and hyperparameters
python train.py --model unet --epochs 100 --batch-size 16 --lr 0.001

# Production mode (advanced settings)
python train.py --mode production

# Quick test (small config for debugging)
python train.py --mode quick_test
```

### Available Command-Line Arguments
```bash
python train.py --help

optional arguments:
  --config CONFIG           Path to config YAML file
  --model {unet,segnet}     Model architecture (default: unet)
  --epochs EPOCHS           Number of epochs (default: 50)
  --batch-size BATCH_SIZE   Batch size (default: 32)
  --lr LR                   Learning rate (default: 0.001)
  --device {cuda,cpu}       Compute device (default: cuda)
  --seed SEED               Random seed (default: 42)
  --mode {default,production,quick_test}  Configuration mode
  --exp-name EXP_NAME       Experiment name (default: semantic_segmentation)
```

---

## Project Structure

```
.
├── data/
│   ├── train_images/           # 2,683 training satellite images
│   ├── train_masks/            # Corresponding training masks
│   ├── valid_images/           # 158 validation images
│   └── valid_masks/            # Corresponding validation masks
│
├── src/                        # Main source code
│   ├── __init__.py
│   ├── config.py              # Configuration system (dataclasses)
│   ├── data.py                # Dataset classes & dataloaders
│   ├── models.py              # Model architectures
│   ├── losses.py              # Loss functions
│   ├── metrics.py             # Evaluation metrics
│   ├── trainer.py             # Training loop
│   ├── utils.py               # Utility functions
│   └── logger.py              # Logging configuration
│
├── models/                     # Trained models & checkpoints
│   └── checkpoints/
│
├── results/                    # Training results & logs
│   ├── logs/                  # TensorBoard logs
│   └── predictions/           # Prediction outputs
│
├── notebook/                   # Jupyter notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Visualisasi.ipynb      # Data Visualization
│   ├── Inference.ipynb        # Inference & Predictions (coming)
│   └── Evaluation.ipynb       # Detailed Evaluation (coming)
│
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Configuration

### Configuration System

The project uses a **hierarchical dataclass-based configuration system** for maximum flexibility:

```python
# src/config.py structure:
Config
├── image: ImageConfig         # Image preprocessing settings
├── training: TrainingConfig   # Training hyperparameters
├── augmentation: AugmentationConfig  # Data augmentation
├── model: ModelConfig         # Model architecture settings
├── loss: LossConfig           # Loss function configuration
├── evaluation: EvaluationConfig
├── device: DeviceConfig       # GPU/CPU settings
├── logging: LoggingConfig
└── checkpoint: CheckpointConfig  # Model checkpointing
```

### Configuration Methods

**Method 1: Command-line Arguments** (Quick experimentation)
```bash
python train.py --epochs 100 --batch-size 16 --lr 0.0005
```

**Method 2: Config YAML File** (Reproducibility)
```bash
# Create custom config file (config_custom.yaml)
python train.py --config config_custom.yaml
```

**Method 3: Pre-defined Modes**
```bash
# Quick test mode (small dataset, few epochs)
python train.py --mode quick_test

# Production mode (full dataset, advanced settings)
python train.py --mode production
```

### Key Configuration Options

**Training:**
```python
config.training.batch_size = 32
config.training.num_epochs = 50
config.training.learning_rate = 1e-3
config.training.optimizer = 'adam'  # or 'adamw', 'sgd'
config.training.scheduler = 'cosine'  # or 'step', 'exponential'
config.training.enable_early_stopping = True
config.training.early_stopping_patience = 10
```

**Data Augmentation:**
```python
config.augmentation.enable_rotation = True
config.augmentation.rotation_degree_range = (-15, 15)
config.augmentation.enable_flip = True
config.augmentation.enable_gaussian_blur = True
config.augmentation.enable_elastic_transform = False
config.augmentation.augmentation_factor = 1.0
```

**Model:**
```python
config.model.model_type = 'unet'
config.model.decoder_channels = (256, 128, 64, 32)
config.model.enable_attention = True
config.model.attention_type = 'scse'  # or 'cbam'
config.model.dropout = 0.2
```

**Loss Function:**
```python
config.loss.loss_type = 'combined'  # dice + bce
config.loss.loss_weights = {'dice': 0.5, 'bce': 0.5}
config.loss.pos_weight = 7.0  # For class imbalance
```

---

## Training

### Training Process

The training pipeline includes:

1. **Data Loading**: Loads train/val/test splits
2. **Augmentation**: Applies augmentation to training data
3. **Model Building**: Instantiates model with config
4. **Training Loop**: Epochs with validation
5. **Checkpointing**: Saves best models and intermediate checkpoints
6. **Early Stopping**: Stops if no improvement
7. **Evaluation**: Final metrics on test set

### Example Training Sessions

**Standard Training (1-2 hours on GPU):**
```bash
python train.py --model unet --epochs 50 --batch-size 32
```

**Large-scale Training (4-8 hours on GPU):**
```bash
python train.py --mode production
```

**Fast Prototyping (5-10 minutes):**
```bash
python train.py --mode quick_test
```

**Custom Configuration:**
```bash
python train.py \
  --model unet \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.0005 \
  --device cuda \
  --exp-name "experiment_v1"
```

### Monitoring Training

**Via Console Output:**
```
Epoch 1/50: 100%|████████| 2146/2146 [6:23<00:00, ...]
  Loss: 0.2345

Validation:
  Loss: 0.1987
  IoU: 0.6789
  Dice: 0.8012
  ...
```

**Via TensorBoard:**
```bash
tensorboard --logdir results/logs
# Then open http://localhost:6006 in browser
```

---

## Evaluation

### Evaluation Metrics

The project computes the following metrics:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **IoU** | TP / (TP + FP + FN) | Intersection over Union (0-1, higher is better) |
| **Dice** | 2×TP / (2×TP + FP + FN) | F1-score for segmentation (0-1, higher is better) |
| **Precision** | TP / (TP + FP) | % of predicted positives that are correct |
| **Recall** | TP / (TP + FN) | % of actual positives that are detected |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean |
| **Sensitivity** | Same as Recall | True Positive Rate |
| **Specificity** | TN / (TN + FP) | True Negative Rate |
| **MCC** | (TP×TN - FP×FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] | Balanced metric for binary classification |

### Multi-Threshold Evaluation

Metrics are computed at multiple thresholds (0.3, 0.4, 0.5, 0.6, 0.7) for detailed analysis:

```
Threshold: 0.5
├── IoU: 0.6234
├── Dice: 0.7654
├── Precision: 0.8123
├── Recall: 0.7456
├── F1: 0.7779
└── ...
```

---

## Results

### Training Output Structure

```
results/
├── logs/
│   └── semantic_segmentation.log      # Detailed log file
├── semantic_segmentation_results.json # Training metrics history
├── semantic_segmentation_config.yaml  # Configuration used
└── predictions/
    ├── 0000_image.jpg                 # Sample predictions
    ├── 0000_prediction.jpg
    └── ...
```

### Example Results

After training, you'll see output like:

```
==========================================
TRAINING COMPLETED SUCCESSFULLY!
==========================================

Test Set Metrics:
  IoU: 0.7234
  DICE: 0.8456
  PRECISION: 0.8789
  RECALL: 0.8123
  F1: 0.8448
  ACCURACY: 0.9567
```

---

## Advanced Usage

### Custom Data Augmentation

Modify `src/config.py`:

```python
config.augmentation.enable_elastic_transform = True
config.augmentation.elastic_alpha = 30
config.augmentation.enable_cutout = True
config.augmentation.cutout_num_holes = 8
config.augmentation.enable_mixup = True
config.augmentation.mixup_alpha = 0.2
```

### Multiple Loss Functions

Use combined loss with custom weights:

```python
config.loss.loss_type = 'combined'
config.loss.loss_weights = {
    'dice': 0.4,
    'bce': 0.4,
    'focal': 0.2
}
config.loss.focal_gamma = 2.0
```

### Learning Rate Scheduling

Different scheduler strategies:

```python
# Cosine annealing
config.training.scheduler = 'cosine'
config.training.scheduler_params = {'T_max': 50}

# Step decay
config.training.scheduler = 'step'
config.training.scheduler_params = {'step_size': 10, 'gamma': 0.5}

# Exponential decay
config.training.scheduler = 'exponential'
config.training.scheduler_params = {'gamma': 0.9}
```

### Mixed Precision Training

Enable for faster training and lower memory:

```python
config.device.mixed_precision = True
```

---

## Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size
config.training.batch_size = 8

# Reduce image size
config.image.img_size = 128

# Enable mixed precision
config.device.mixed_precision = True
```

### Slow Training
```python
# Use lighter model
config.model.model_type = 'segnet'

# Reduce image size
config.image.img_size = 128

# Use more workers
config.device.num_workers = 8
```

### Poor Convergence
```python
# Adjust learning rate
config.training.learning_rate = 5e-4

# Add more augmentation
config.augmentation.augmentation_factor = 2.0

# Reduce dropout
config.model.dropout = 0.1
```

### Data Not Found
```bash
# Ensure data structure:
data/
├── train_images/    (2,683 JPG files)
├── train_masks/     (2,683 PNG files)
├── valid_images/    (158 JPG files)
└── valid_masks/     (158 PNG files)
```

---

## Dataset Details

### Dataset Statistics
- **Training Images**: 2,683 (94.5%)
- **Validation Images**: 158 (5.5%)
- **Total Samples**: 2,841
- **Image Size**: 512×512 pixels
- **Format**: RGB JPG images with grayscale PNG masks
- **Class Imbalance**: ~7.2:1 (background:water)

### Preprocessing
- Images resized to 256×256 (configurable)
- Normalized using ImageNet statistics
- Masks binarized at threshold > 127

---

## References

### Papers
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.02674)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/abs/1706.05721)

### Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Albumentations Library](https://albumentations.ai/)
- [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [Intersection over Union (IoU)](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit PRs or open issues for bugs and feature requests.

---

## Checklist

- [x] PyTorch implementation
- [x] U-Net and SegNet architectures
- [x] Multiple loss functions
- [x] Data augmentation pipeline
- [x] Configuration system
- [x] Training loop with early stopping
- [x] Comprehensive evaluation metrics
- [ ] Inference notebook
- [ ] Evaluation notebook
- [ ] DeepLabV3/V3+ architecture
- [ ] ONNX export
- [ ] Web API deployment

---

**Last Updated**: March 2026  
**Status**: Production Ready
