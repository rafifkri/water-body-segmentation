"""
Training loop and trainer class
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import logging
import time
from tqdm.auto import tqdm
import numpy as np

from .config import Config
from .models import create_model
from .losses import create_loss
from .metrics import SegmentationMetrics, AverageMeter
from .utils import (
    save_checkpoint, clip_gradients, get_device,
    create_optimizer, create_scheduler, print_model_info
)


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        config: Config,
        device: Optional[torch.device] = None
    ):

        self.config = config
        self.device = device or get_device(config.device.device)
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = self._build_loss()
        self.loss_fn.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Metrics
        self.metrics = SegmentationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.train_history = {'loss': [], 'metrics': []}
        self.val_history = {'loss': [], 'metrics': []}
    
    def _build_model(self) -> nn.Module:
        """Build model"""
        logger.info(f"Building {self.config.model.model_type} model...")
        
        model = create_model(
            model_type=self.config.model.model_type,
            in_channels=self.config.image.img_channels,
            num_classes=self.config.model.num_classes,
            decoder_channels=self.config.model.decoder_channels,
            dropout=self.config.model.dropout,
            attention=self.config.model.enable_attention,
            attention_type=self.config.model.attention_type,
            normalization=self.config.model.normalization,
            activation=self.config.model.activation
        )
        
        print_model_info(model)
        return model
    
    def _build_loss(self) -> nn.Module:
        """Build loss function"""
        logger.info(f"Building {self.config.loss.loss_type} loss...")
        
        loss_fn = create_loss(
            loss_type=self.config.loss.loss_type,
            loss_weights=self.config.loss.loss_weights,
            dice_smooth=self.config.loss.dice_smooth,
            focal_alpha=self.config.loss.focal_alpha,
            focal_gamma=self.config.loss.focal_gamma,
            pos_weight=self.config.loss.pos_weight
        )
        
        return loss_fn
    
    def prepare_for_training(self) -> None:
        """Prepare optimizer and scheduler for training"""
        logger.info("Preparing for training...")
        
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            optimizer_type=self.config.training.optimizer,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            momentum=self.config.training.momentum,
            **(self.config.training.optimizer_params or {})
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.config.training.scheduler,
            **(self.config.training.scheduler_params or {})
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:

        self.model.train()
        
        loss_meter = AverageMeter('Loss')
        predictions_list = []
        targets_list = []
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs} [TRAIN]",
            total=len(train_loader),
            bar_format='{l_bar}{bar:30}{r_bar} {percentage:3.0f}% [{elapsed}<{remaining}]'
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.loss_fn(predictions, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                grad_norm = clip_gradients(self.model, self.config.training.gradient_clip_val)
            
            # Optimization step
            self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.shape[0])
            
            # Collect predictions for metrics calculation
            predictions_list.append(torch.sigmoid(predictions).detach().cpu())
            targets_list.append(masks.detach().cpu())
            
            # Update progress bar every 5 batches with running metrics
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                metrics_dict = {'loss': f'{loss_meter.avg:.4f}'}
                progress_bar.set_postfix(metrics_dict)
        
        # Compute full metrics for epoch
        if predictions_list:
            predictions_tensor = torch.cat(predictions_list, dim=0)
            targets_tensor = torch.cat(targets_list, dim=0)
            
            epoch_metrics = self.metrics.compute_metrics(
                predictions_tensor,
                targets_tensor,
                threshold=self.config.evaluation.prediction_threshold,
                metrics_list=self.config.evaluation.metrics
            )
            epoch_metrics['loss'] = loss_meter.avg
        else:
            epoch_metrics = {'loss': loss_meter.avg}
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:

        self.model.eval()
        
        loss_meter = AverageMeter('Loss')
        predictions_list = []
        targets_list = []
        
        progress_bar = tqdm(val_loader, desc="  [VALIDATE]", total=len(val_loader))
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)
                
                # Collect for metrics
                predictions_list.append(torch.sigmoid(predictions).cpu())
                targets_list.append(masks.cpu())
                
                # Update loss
                loss_meter.update(loss.item(), images.shape[0])
                
                progress_bar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}'
                })
        
        # Compute metrics
        predictions_tensor = torch.cat(predictions_list, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)
        
        metrics = self.metrics.compute_metrics(
            predictions_tensor,
            targets_tensor,
            threshold=self.config.evaluation.prediction_threshold,
            metrics_list=self.config.evaluation.metrics
        )
        
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:

        logger.info("="*80)
        logger.info("STARTING TRAINING".center(80))
        logger.info("="*80)
        
        # Prepare for training
        self.prepare_for_training()
        
        # Early stopping variables
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_metrics['loss'])
            
            # Log training metrics
            logger.info(f"\n  📊 TRAINING METRICS:")
            logger.info(f"    Loss:     {train_metrics['loss']:.4f}")
            for metric_name, metric_value in sorted(train_metrics.items()):
                if metric_name != 'loss':
                    logger.info(f"    {metric_name.upper():10s}: {metric_value:.4f}")
            
            # Validate
            if (epoch + 1) % self.config.training.validate_every == 0:
                val_metrics = self.validate(val_loader)
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['metrics'].append(val_metrics)
                
                # Log validation metrics
                logger.info(f"\n  📊 VALIDATION METRICS:")
                logger.info(f"    Loss:     {val_metrics['loss']:.4f}")
                for metric_name, metric_value in sorted(val_metrics.items()):
                    if metric_name != 'loss':
                        logger.info(f"    {metric_name.upper():10s}: {metric_value:.4f}")
                
                # Save checkpoint
                if self.config.checkpoint.save_checkpoints:
                    if (epoch + 1) % self.config.checkpoint.save_every_n_epochs == 0:
                        checkpoint_path = self.config.checkpoint.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            val_metrics,
                            checkpoint_path
                        )
                        logger.info(f"  ✅ Checkpoint saved: {checkpoint_path.name}")
                
                # Early stopping
                if self.config.training.enable_early_stopping:
                    monitor_metric = self.config.checkpoint.save_best_metric
                    if monitor_metric in val_metrics:
                        current_value = val_metrics[monitor_metric]
                        
                        if monitor_metric not in self.best_metrics or \
                           (self.config.checkpoint.save_best_mode == 'max' and current_value > self.best_metrics[monitor_metric]) or \
                           (self.config.checkpoint.save_best_mode == 'min' and current_value < self.best_metrics[monitor_metric]):
                            
                            self.best_metrics[monitor_metric] = current_value
                            patience_counter = 0
                            
                            # Save best model
                            best_path = self.config.checkpoint.checkpoint_dir / "best_model.pt"
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                epoch,
                                val_metrics,
                                best_path
                            )
                            logger.info(f"  ⭐ Best model saved! {monitor_metric}: {current_value:.4f}")
                        else:
                            patience_counter += 1
                            logger.info(f"  ⚠️  No improvement. Patience: {patience_counter}/{self.config.training.early_stopping_patience}")
                            
                            if patience_counter >= self.config.training.early_stopping_patience:
                                logger.info(f"\n  🛑 Early stopping triggered after {epoch + 1} epochs")
                                break
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED".center(80))
        logger.info("="*80)
        logger.info(f"Total time: {elapsed_time / 3600:.2f} hours ({elapsed_time / 60:.1f} minutes)")
        if self.best_metrics:
            logger.info(f"Best metrics: {self.best_metrics}")
        logger.info("="*80 + "\n")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'elapsed_time': elapsed_time
        }


logger.info("Trainer module loaded")
