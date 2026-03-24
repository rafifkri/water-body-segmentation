"""
Data loading and preprocessing module
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from .config import Config


logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        img_size: int = 256,
        transforms: Optional[A.Compose] = None,
        mask_threshold: float = 127.5,
        normalize: bool = True,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        return_masks: bool = True
    ):

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.transforms = transforms
        self.mask_threshold = mask_threshold
        self.normalize = normalize
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.return_masks = return_masks
        
        # Get image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Load mask if available and requested
        mask = None
        if self.return_masks:
            mask_path = self.masks_dir / f"{img_name}{img_path.suffix}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                # Binary mask
                mask = (mask > self.mask_threshold).astype(np.uint8)
            else:
                # If mask not found, create empty mask
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                logger.warning(f"Mask not found for {img_name}, using empty mask")
        
        # Apply transforms
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            if self.return_masks:
                mask_output = augmented['mask']
                # Check if mask is already a tensor (from ToTensorV2)
                if isinstance(mask_output, torch.Tensor):
                    # Add channel dimension if needed
                    if mask_output.ndim == 2:
                        mask = mask_output.unsqueeze(0).float()
                    else:
                        mask = mask_output.float()
                else:
                    # Convert from numpy
                    mask = torch.from_numpy(mask_output).unsqueeze(0).float()
        else:
            # Manual normalization
            image = image.astype(np.float32) / 255.0
            if self.normalize:
                image = self._normalize(image)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            if self.return_masks:
                mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        result = {'image': image}
        if self.return_masks:
            result['mask'] = mask
        
        return result
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization to image"""
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.normalize_mean[i]) / self.normalize_std[i]
        return image


def get_transforms(config: Config, mode: str = 'train') -> A.Compose:

    aug_config = config.augmentation
    img_size = config.image.img_size
    normalize_mean = config.image.normalize_mean
    normalize_std = config.image.normalize_std
    
    transforms_list = []
    
    if mode == 'train' and aug_config.enable_augmentation:
        # Geometric augmentations
        if aug_config.enable_rotation:
            transforms_list.append(
                A.Rotate(
                    limit=aug_config.rotation_degree_range,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                )
            )
        
        if aug_config.enable_flip:
            transforms_list.append(A.HorizontalFlip(p=aug_config.flip_prob))
            transforms_list.append(A.VerticalFlip(p=aug_config.flip_prob))
        
        # Color augmentations
        if aug_config.enable_brightness:
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=[-20, 20],
                    contrast_limit=[-20, 20],
                    p=0.5
                )
            )
        
        if aug_config.enable_contrast:
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=[-10, 10],
                    contrast_limit=[-10, 10],
                    p=0.3
                )
            )
        
        if aug_config.enable_gaussian_blur:
            transforms_list.append(
                A.GaussianBlur(blur_limit=(3, 5), p=0.3)
            )
        
        if aug_config.enable_elastic_transform:
            transforms_list.append(
                A.ElasticTransform(
                    alpha=aug_config.elastic_alpha,
                    sigma=aug_config.elastic_sigma,
                    p=0.3
                )
            )
        
        if aug_config.enable_gaussian_noise:
            transforms_list.append(
                A.GaussNoise(p=0.2)
            )
        
        if aug_config.enable_cutout:
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=aug_config.cutout_num_holes,
                    max_height=aug_config.cutout_hole_size[0],
                    max_width=aug_config.cutout_hole_size[1],
                    p=aug_config.cutout_prob
                )
            )
    
    # Normalization
    transforms_list.append(
        A.Normalize(mean=normalize_mean, std=normalize_std)
    )
    
    # Convert to tensor
    transforms_list.append(ToTensorV2())
    
    return A.Compose(
        transforms_list,
        is_check_shapes=False
    )


def create_dataloaders(
    config: Config,
    train_val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Determine base data path
    base_path = Path.cwd() / 'data'
    
    # Get transforms
    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')
    
    # Load training dataset
    logger.info("Loading training dataset...")
    train_dataset = SegmentationDataset(
        images_dir=base_path / 'train_images',
        masks_dir=base_path / 'train_masks',
        img_size=config.image.img_size,
        transforms=train_transforms,
        mask_threshold=config.image.mask_threshold,
        normalize_mean=config.image.normalize_mean,
        normalize_std=config.image.normalize_std
    )
    
    # Load validation dataset
    logger.info("Loading validation dataset...")
    valid_dataset = SegmentationDataset(
        images_dir=base_path / 'valid_images',
        masks_dir=base_path / 'valid_masks',
        img_size=config.image.img_size,
        transforms=val_transforms,
        mask_threshold=config.image.mask_threshold,
        normalize_mean=config.image.normalize_mean,
        normalize_std=config.image.normalize_std
    )
    
    # Split training data
    train_size = int(len(train_dataset) * (1 - train_val_split))
    val_size = len(train_dataset) - train_size
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.random_seed)
    )
    
    logger.info(f"Train: {len(train_subset)}, Val (from train): {len(val_subset)}, Test: {len(valid_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory
    )
    
    test_loader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_stats(dataset: Dataset, num_samples: int = 100) -> Dict[str, Any]:

    stats = {
        'num_samples': len(dataset),
        'mean_water_ratio': 0,
        'std_water_ratio': 0,
        'images_with_water': 0,
        'images_without_water': 0
    }
    
    if num_samples > len(dataset):
        num_samples = len(dataset)
    
    water_ratios = []
    for i in range(num_samples):
        sample = dataset[i]
        mask = sample['mask']
        water_ratio = mask.float().mean().item()
        water_ratios.append(water_ratio)
        
        if water_ratio > 0:
            stats['images_with_water'] += 1
        else:
            stats['images_without_water'] += 1
    
    stats['mean_water_ratio'] = np.mean(water_ratios)
    stats['std_water_ratio'] = np.std(water_ratios)
    
    return stats
