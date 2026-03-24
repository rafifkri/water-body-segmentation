# Configuration for Semantic Segmentation Project
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data paths
TRAIN_IMAGES_PATH = DATA_DIR / 'train_images'
TRAIN_MASKS_PATH = DATA_DIR / 'train_masks'
VALID_IMAGES_PATH = DATA_DIR / 'valid_images'
VALID_MASKS_PATH = DATA_DIR / 'valid_masks'

# Image Configuration
IMG_SIZE = 256
IMG_CHANNELS = 3

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

# Data Augmentation
ENABLE_AUGMENTATION = True
AUGMENTATION_FACTOR = 1.0  # Multiply training data by this factor

# Model Configuration
MODEL_TYPE = 'unet'  # Options: 'unet', 'deeplabv3'
LOSS_FUNCTION = 'combined'  # Options: 'dice', 'bce', 'combined'

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Evaluation
PREDICTION_THRESHOLD = 0.5
SAVE_PREDICTIONS = True

# Random Seed
RANDOM_SEED = 42

# Logging
VERBOSE = 1
SAVE_FREQ = 5  # Save model every N epochs
