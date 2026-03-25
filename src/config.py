"""
Configuration module for semantic segmentation training
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch
import logging


# ===========================
# PATH CONFIGURATION
# ===========================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = RESULTS_DIR / 'logs'
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset paths
TRAIN_IMAGES_PATH = DATA_DIR / 'train_images'
TRAIN_MASKS_PATH = DATA_DIR / 'train_masks'
VALID_IMAGES_PATH = DATA_DIR / 'valid_images'
VALID_MASKS_PATH = DATA_DIR / 'valid_masks'


# ===========================
# IMAGE CONFIGURATION
# ===========================
@dataclass
class ImageConfig:
    """Image preprocessing configuration"""
    img_size: int = 256
    img_channels: int = 3
    mask_channels: int = 1
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    mask_threshold: float = 127.5
    interpolation: str = 'bilinear'  # Options: 'bilinear', 'nearest'


# ===========================
# TRAINING CONFIGURATION
# ===========================
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer configuration
    optimizer: str = 'adam'  # Options: 'adam', 'adamw', 'sgd'
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Learning rate scheduler
    scheduler: str = 'cosine'  # Options: 'cosine', 'step', 'exponential', 'linear'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {'T_max': 50})
    
    # Validation
    validation_split: float = 0.2
    validate_every: int = 1  # Validate every N epochs
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Gradient
    gradient_clip_val: float = 1.0
    enable_gradient_accumulation: bool = False
    accumulation_steps: int = 4
    
    # Random seed
    random_seed: int = 42


# ===========================
# DATA AUGMENTATION
# ===========================
@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enable_augmentation: bool = True
    
    # Geometric augmentations
    enable_rotation: bool = True
    rotation_degree_range: tuple = (-15, 15)
    
    enable_flip: bool = True
    flip_prob: float = 0.5
    flip_directions: List[str] = field(default_factory=lambda: ['horizontal', 'vertical'])
    
    # Color augmentations
    enable_brightness: bool = True
    brightness_range: tuple = (0.8, 1.2)
    
    enable_contrast: bool = True
    contrast_range: tuple = (0.8, 1.2)
    
    enable_saturation: bool = True
    saturation_range: tuple = (0.8, 1.2)
    
    enable_hue: bool = False
    hue_shift_limit: int = 20
    
    enable_elastic_transform: bool = False
    elastic_alpha: int = 30
    elastic_sigma: int = 5
    
    enable_gaussian_blur: bool = True
    blur_kernel_range: tuple = (3, 7)
    
    enable_gaussian_noise: bool = False
    gaussian_noise_std: float = 0.01
    
    # Cutout/Mixup
    enable_cutout: bool = False
    cutout_prob: float = 0.5
    cutout_num_holes: int = 8
    cutout_hole_size: tuple = (8, 8)
    
    enable_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # Advanced
    augmentation_p: float = 0.8  # Probability to apply augmentation
    augmentation_factor: float = 1.0  # Multiply dataset by this factor


# ===========================
# MODEL CONFIGURATION
# ===========================
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = 'unet'  # Options: 'unet', 'unet_resnet', 'deeplabv3', 'deeplabv3plus', 'segnet', 'fcn'
    encoder_name: str = 'resnet50'  # For models with encoders
    encoder_weights: str = 'imagenet'  # 'imagenet' or None
    decoder_channels: tuple = (512, 256, 128, 64, 32)
    num_classes: int = 1  # Binary segmentation
    in_channels: int = 3
    
    # Attention mechanisms
    enable_attention: bool = True
    attention_type: str = 'scse'  # Options: 'scse', 'cbam'
    
    # Dropout
    dropout: float = 0.2
    
    # Layer normalization
    normalization: str = 'batch'  # Options: 'batch', 'instance', 'layer'
    
    # Activation
    activation: str = 'relu'  # Options: 'relu', 'gelu', 'swish'


# ===========================
# LOSS CONFIGURATION
# ===========================
@dataclass
class LossConfig:
    """Loss function configuration"""
    loss_type: str = 'combined'  # Options: 'dice', 'bce', 'focal', 'combined', 'weighted_bce'
    
    # Loss weights (if using multiple losses)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'dice': 0.5,
        'bce': 0.5,
        'focal': 0.0
    })
    
    # Dice loss params
    dice_smooth: float = 1.0
    dice_power: int = 2
    
    # Focal loss params
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Weighted BCE params
    pos_weight: float = 7.0  # Class imbalance weight
    
    # Class weights
    class_weights: List[float] = field(default_factory=lambda: [1.0, 7.0])


# ===========================
# EVALUATION CONFIGURATION
# ===========================
@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    prediction_threshold: float = 0.5
    save_predictions: bool = True
    save_prediction_format: str = 'png'  # Options: 'png', 'jpg', 'tiff'
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        'iou', 'dice', 'precision', 'recall', 'f1', 'accuracy',
        'specificity', 'sensitivity', 'mcc'
    ])
    
    # Multi-threshold evaluation
    evaluate_at_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])


# ===========================
# DEVICE CONFIGURATION
# ===========================
@dataclass
class DeviceConfig:
    """Device and computation configuration"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    benchmark: bool = True  # Use cudnn benchmark for faster training
    deterministic: bool = False  # Set for reproducibility (slower)
    mixed_precision: bool = False  # AMP (Automatic Mixed Precision)
    dtype: torch.dtype = torch.float32


# ===========================
# LOGGING CONFIGURATION
# ===========================
@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    log_to_file: bool = True
    log_dir: Path = LOGS_DIR
    log_name: str = 'training'
    log_every_n_steps: int = 10
    save_wandb: bool = False  # Weights & Biases integration
    wandb_project: str = 'semantic-segmentation'
    wandb_entity: str = None


# ===========================
# CHECKPOINT CONFIGURATION
# ===========================
@dataclass
class CheckpointConfig:
    """Model checkpoint configuration"""
    save_checkpoints: bool = True
    checkpoint_dir: Path = CHECKPOINTS_DIR
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    save_best_metric: str = 'val_dice'  # Metric to monitor for best model
    save_best_mode: str = 'max'  # 'max' or 'min'
    keep_last_n_checkpoints: int = 3  # Delete older checkpoints
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True


# ===========================
# COMPLETE CONFIGURATION
# ===========================
@dataclass
class Config:
    """Main configuration class"""
    # Sub-configurations
    image: ImageConfig = field(default_factory=ImageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Experiment name
    experiment_name: str = 'semantic_segmentation'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def asdict_recursive(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [asdict_recursive(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: asdict_recursive(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: asdict_recursive(v) for k, v in obj.__dict__.items()}
            else:
                return str(obj)
        
        return asdict_recursive(self)
    
    def save_yaml(self, path: Path) -> None:
        """Save configuration to YAML file"""
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        except ImportError:
            logging.warning("PyYAML not installed. Skipping YAML config save.")
    
    def save_json(self, path: Path) -> None:
        """Save configuration to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ===========================
# DEFAULT CONFIGURATIONS
# ===========================
def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def get_config_for_quick_test() -> Config:
    """Get configuration for quick testing"""
    config = Config()
    config.training.batch_size = 8
    config.training.num_epochs = 5
    config.image.img_size = 128
    config.model.model_type = 'unet'
    config.training.enable_early_stopping = False
    return config


def get_config_for_production() -> Config:
    """Get configuration for production training"""
    config = Config()
    config.training.batch_size = 32
    config.training.num_epochs = 100
    config.training.scheduler = 'cosine'
    config.augmentation.augmentation_factor = 2.0
    config.device.mixed_precision = True
    config.device.benchmark = True
    config.checkpoint.save_best_only = True
    config.evaluation.evaluate_at_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    return config
