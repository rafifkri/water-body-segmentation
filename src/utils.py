"""
Utility functions for training and evaluation
"""
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json
from datetime import datetime


logger = logging.getLogger(__name__)


# ===========================
# RANDOM SEED MANAGEMENT
# ===========================
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ===========================
# DEVICE MANAGEMENT
# ===========================
def get_device(device: str = 'cuda') -> torch.device:
    """
    Get device (GPU or CPU)
    
    Args:
        device: 'cuda' or 'cpu'
    
    Returns:
        torch.device object
    """
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = 'cpu'
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


# ===========================
# MODEL CHECKPOINTING
# ===========================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: Path,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint
        device: Device to load checkpoint to
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    return checkpoint


# ===========================
# LEARNING RATE SCHEDULER
# ===========================
def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        **kwargs: Additional optimizer arguments
    
    Returns:
        PyTorch optimizer
    """
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"Optimizer: {optimizer_type}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler arguments
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **kwargs
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **kwargs
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            **kwargs
        )
    elif scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            **kwargs
        )
    else:
        logger.warning(f"Unknown scheduler: {scheduler_type}, using None")
        return None
    
    logger.info(f"Scheduler: {scheduler_type}")
    return scheduler


# ===========================
# GRADIENT MANAGEMENT
# ===========================
def clip_gradients(
    model: nn.Module,
    clip_value: float = 1.0
) -> float:
    """
    Clip gradients for stability
    
    Args:
        model: PyTorch model
        clip_value: Maximum gradient norm
    
    Returns:
        Total gradient norm
    """
    return nn.utils.clip_grad_norm_(model.parameters(), clip_value)


# ===========================
# MODEL ANALYSIS
# ===========================
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters"""
    return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module) -> None:
    """Print model information"""
    total_params = count_total_parameters(model)
    trainable_params = count_parameters(model)
    
    logger.info("="*60)
    logger.info("MODEL INFORMATION")
    logger.info("="*60)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable params: {total_params - trainable_params:,}")
    logger.info("="*60)


# ===========================
# MIXED PRECISION TRAINING
# ===========================
def create_scaler() -> torch.cuda.amp.GradScaler:
    """Create gradient scaler for mixed precision"""
    return torch.cuda.amp.GradScaler()


# ===========================
# RESULTS SAVING
# ===========================
def save_results(
    results: Dict[str, Any],
    save_path: Path,
    format: str = 'json'
) -> None:
    """
    Save results to file
    
    Args:
        results: Results dictionary
        save_path: Path to save
        format: 'json' or 'txt'
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'txt':
        with open(save_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Results saved to {save_path}")


# ===========================
# VISUALIZATION UTILITIES
# ===========================
def save_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    save_dir: Path,
    indices: list = None,
    num_samples: int = 5
) -> None:
    """
    Save prediction visualizations
    
    Args:
        images: Image array
        predictions: Prediction array
        targets: Target array
        save_dir: Directory to save
        indices: Indices to save (None = random)
        num_samples: Number of samples to save
    """
    import cv2
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if indices is None:
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for idx in indices:
        # Prepare image (assume normalized to [0, 1])
        img = (images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Prepare prediction and target
        pred = (predictions[idx].squeeze() * 255).astype(np.uint8)
        target = (targets[idx].squeeze() * 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(save_dir / f"{idx:04d}_image.jpg"), img)
        cv2.imwrite(str(save_dir / f"{idx:04d}_prediction.jpg"), pred)
        cv2.imwrite(str(save_dir / f"{idx:04d}_target.jpg"), target)
    
    logger.info(f"Saved {len(indices)} predictions to {save_dir}")


logger.info("Utilities module loaded")
