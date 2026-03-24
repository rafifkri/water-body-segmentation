"""
Utility functions for data loading, preprocessing, and metrics calculation
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
import tensorflow as tf
from config import IMG_SIZE, IMG_CHANNELS, RANDOM_SEED


def load_image(image_path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        size: Output size (size x size)
    
    Returns:
        Normalized image array [0, 1] with shape (size, size, 3)
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype('float32') / 255.0
    return img


def load_mask(mask_path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    Load and preprocess mask
    
    Args:
        mask_path: Path to mask file
        size: Output size (size x size)
    
    Returns:
        Binary mask array with shape (size, size)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size))
    mask = (mask > 127).astype('float32')
    return mask


def load_dataset(
    images_path: Path,
    masks_path: Path,
    size: int = IMG_SIZE,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load complete dataset
    
    Args:
        images_path: Path to images directory
        masks_path: Path to masks directory
        size: Output image size
        verbose: Print loading progress
    
    Returns:
        Tuple of (images array, masks array)
    """
    image_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    mask_files = sorted([f for f in masks_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    assert len(image_files) == len(mask_files), "Number of images and masks don't match"
    
    images = []
    masks = []
    
    for i, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(image_files)} samples...")
        
        img = load_image(str(img_path), size=size)
        mask = load_mask(str(mask_path), size=size)
        
        images.append(img)
        masks.append(mask)
    
    images = np.array(images, dtype='float32')
    masks = np.array(masks, dtype='float32')
    
    return images, masks


def augment_images(
    images: np.ndarray,
    masks: np.ndarray,
    num_augmentations: int = 1,
    seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to images and masks
    
    Args:
        images: Array of images
        masks: Array of masks
        num_augmentations: Number of augmented versions to create
        seed: Random seed
    
    Returns:
        Augmented images and masks
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    augmented_images = [images.copy()]
    augmented_masks = [masks.copy()]
    
    for aug_idx in range(num_augmentations):
        aug_imgs = []
        aug_masks = []
        
        for img, mask in zip(images, masks):
            # Random rotation
            angle = np.random.uniform(-15, 15)
            img_rot = tf.keras.preprocessing.image.random_rotation(
                np.expand_dims(img, 0), angle, seed=seed + aug_idx
            )[0]
            mask_rot = tf.keras.preprocessing.image.random_rotation(
                np.expand_dims(np.expand_dims(mask, -1), 0), angle, seed=seed + aug_idx
            )[0, :, :, 0]
            
            # Random flip (horizontal)
            if np.random.rand() > 0.5:
                img_rot = np.fliplr(img_rot)
                mask_rot = np.fliplr(mask_rot)
            
            # Random flip (vertical)
            if np.random.rand() > 0.5:
                img_rot = np.flipud(img_rot)
                mask_rot = np.flipud(mask_rot)
            
            aug_imgs.append(img_rot)
            aug_masks.append(mask_rot)
        
        augmented_images.append(np.array(aug_imgs, dtype='float32'))
        augmented_masks.append(np.array(aug_masks, dtype='float32'))
    
    # Concatenate all augmented data
    X_aug = np.concatenate(augmented_images, axis=0)
    y_aug = np.concatenate(augmented_masks, axis=0)
    
    return X_aug, y_aug


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate segmentation metrics
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks (probabilities)
        threshold: Binarization threshold
    
    Returns:
        Dictionary of metrics
    """
    y_pred_binary = (y_pred > threshold).astype('float32')
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Confusion matrix
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    
    # Metrics
    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    
    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn)
    }


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1):
    """IoU (Intersection over Union) metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    """Combined Dice + Binary Crossentropy loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


def print_dataset_info(X_train: np.ndarray, y_train: np.ndarray,
                       X_valid: np.ndarray, y_valid: np.ndarray) -> None:
    """Print dataset information"""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Training images shape: {X_train.shape}")
    print(f"Training masks shape: {y_train.shape}")
    print(f"Validation images shape: {X_valid.shape}")
    print(f"Validation masks shape: {y_valid.shape}")
    print(f"\nValue ranges:")
    print(f"  Images: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Masks: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print("="*60 + "\n")
