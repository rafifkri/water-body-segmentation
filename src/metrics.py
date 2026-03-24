"""
Evaluation metrics for semantic segmentation
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import logging


logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Class for computing segmentation metrics
    """
    
    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        metrics_list: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute multiple segmentation metrics
        
        Args:
            predictions: (B, 1, H, W) predictions (probabilities or logits)
            targets: (B, 1, H, W) binary targets
            threshold: Classification threshold
            metrics_list: List of metrics to compute
        
        Returns:
            Dictionary of metrics
        """
        if metrics_list is None:
            metrics_list = ['iou', 'dice', 'precision', 'recall', 'f1', 'accuracy']
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Binarize predictions
        predictions_binary = (predictions > threshold).astype(np.float32)
        
        # Flatten
        predictions_flat = predictions_binary.flatten()
        targets_flat = targets.flatten()
        
        # Confusion matrix
        tp = np.sum((targets_flat == 1) & (predictions_flat == 1))
        fp = np.sum((targets_flat == 0) & (predictions_flat == 1))
        fn = np.sum((targets_flat == 1) & (predictions_flat == 0))
        tn = np.sum((targets_flat == 0) & (predictions_flat == 0))
        
        metrics = {}
        
        # IoU (Intersection over Union)
        if 'iou' in metrics_list:
            iou = tp / (tp + fp + fn + 1e-7)
            metrics['iou'] = iou
        
        # Dice Coefficient
        if 'dice' in metrics_list:
            dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
            metrics['dice'] = dice
        
        # Precision
        if 'precision' in metrics_list:
            precision = tp / (tp + fp + 1e-7)
            metrics['precision'] = precision
        
        # Recall/Sensitivity
        if 'recall' in metrics_list:
            recall = tp / (tp + fn + 1e-7)
            metrics['recall'] = recall
            metrics['sensitivity'] = recall
        
        # F1-Score
        if 'f1' in metrics_list:
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            metrics['f1'] = f1
        
        # Accuracy
        if 'accuracy' in metrics_list:
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
            metrics['accuracy'] = accuracy
        
        # Specificity
        if 'specificity' in metrics_list:
            specificity = tn / (tn + fp + 1e-7)
            metrics['specificity'] = specificity
        
        # Matthew's Correlation Coefficient
        if 'mcc' in metrics_list:
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7)
            metrics['mcc'] = mcc
        
        return metrics
    
    @staticmethod
    def compute_metrics_per_image(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Compute metrics for each image separately
        
        Args:
            predictions: (B, 1, H, W) predictions
            targets: (B, 1, H, W) targets
            threshold: Classification threshold
        
        Returns:
            Tuple of (per-image metrics, aggregated metrics)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions_binary = (predictions > threshold).astype(np.float32)
        
        batch_size = predictions.shape[0]
        per_image_metrics = {
            'iou': np.zeros(batch_size),
            'dice': np.zeros(batch_size),
            'precision': np.zeros(batch_size),
            'recall': np.zeros(batch_size),
            'f1': np.zeros(batch_size)
        }
        
        for i in range(batch_size):
            pred = predictions_binary[i].flatten()
            target = targets[i].flatten()
            
            tp = np.sum((target == 1) & (pred == 1))
            fp = np.sum((target == 0) & (pred == 1))
            fn = np.sum((target == 1) & (pred == 0))
            
            per_image_metrics['iou'][i] = tp / (tp + fp + fn + 1e-7)
            per_image_metrics['dice'][i] = 2 * tp / (2 * tp + fp + fn + 1e-7)
            per_image_metrics['precision'][i] = tp / (tp + fp + 1e-7)
            per_image_metrics['recall'][i] = tp / (tp + fn + 1e-7)
            per_image_metrics['f1'][i] = 2 * per_image_metrics['precision'][i] * per_image_metrics['recall'][i] / (per_image_metrics['precision'][i] + per_image_metrics['recall'][i] + 1e-7)
        
        # Aggregate metrics
        aggregated_metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in per_image_metrics.items()
        }
        
        return per_image_metrics, aggregated_metrics
    
    @staticmethod
    def compute_confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Compute confusion matrix elements
        
        Args:
            predictions: (B, 1, H, W) predictions
            targets: (B, 1, H, W) targets
            threshold: Classification threshold
        
        Returns:
            Dictionary with TP, FP, FN, TN
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions_binary = (predictions > threshold).astype(np.float32)
        
        predictions_flat = predictions_binary.flatten()
        targets_flat = targets.flatten()
        
        tp = int(np.sum((targets_flat == 1) & (predictions_flat == 1)))
        fp = int(np.sum((targets_flat == 0) & (predictions_flat == 1)))
        fn = int(np.sum((targets_flat == 1) & (predictions_flat == 0)))
        tn = int(np.sum((targets_flat == 0) & (predictions_flat == 0)))
        
        return {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }


# ===========================
# METRIC CLASSES
# ===========================
class IoUMetric:
    """Intersection over Union (IoU) metric"""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions_binary = (predictions > self.threshold).float()
        
        tp = (predictions_binary * targets).sum().item()
        fp = (predictions_binary * (1 - targets)).sum().item()
        fn = ((1 - predictions_binary) * targets).sum().item()
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
    
    def compute(self) -> float:
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        return iou
    
    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0


class DiceMetric:
    """Dice coefficient metric"""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions_binary = (predictions > self.threshold).float()
        
        tp = (predictions_binary * targets).sum().item()
        fp = (predictions_binary * (1 - targets)).sum().item()
        fn = ((1 - predictions_binary) * targets).sum().item()
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
    
    def compute(self) -> float:
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-7)
        return dice
    
    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0


class AverageMeter:
    """Computes and stores average and current value"""
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


logger.info("Metrics module loaded")
