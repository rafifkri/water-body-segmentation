#!/usr/bin/env python
"""
Main training script for semantic segmentation
"""
import argparse
import logging
from pathlib import Path
import json
import yaml

from .config import get_default_config, get_config_for_production, get_config_for_quick_test
from .data import create_dataloaders
from .trainer import Trainer
from .logger import setup_logging
from .utils import set_seed


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train semantic segmentation model for water body detection'
    )
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'segnet', 'deeplabv3', 'deeplabv3plus'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--mode', type=str, default='default',
                       choices=['default', 'production', 'quick_test'],
                       help='Configuration mode')
    parser.add_argument('--exp-name', type=str, default='semantic_segmentation',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        # Load from YAML
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # TODO: Convert dict to Config object
        config = get_default_config()
    elif args.mode == 'production':
        config = get_config_for_production()
    elif args.mode == 'quick_test':
        config = get_config_for_quick_test()
    else:
        config = get_default_config()
    
    # Override config with arguments
    config.experiment_name = args.exp_name
    if args.model:
        config.model.model_type = args.model
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    config.device.device = args.device
    config.training.random_seed = args.seed
    
    # Setup logging
    logger = setup_logging(
        config.logging.log_dir,
        log_name=config.experiment_name,
        log_level=config.logging.log_level,
        log_to_file=config.logging.log_to_file
    )
    
    # Log configuration
    logger.info("="*80)
    logger.info("SEMANTIC SEGMENTATION - WATER BODY DETECTION")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {config.device.device}")
    logger.info(f"Seed: {config.training.random_seed}")
    
    # Set seed
    set_seed(config.training.random_seed)
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(config)
    
    # Train
    logger.info("\nStarting training...")
    training_results = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    training_results['test_metrics'] = test_metrics
    
    # Save results
    logger.info("\nSaving results...")
    results_path = config.results_dir / f'{config.experiment_name}_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    
    # Save config
    config_path = config.results_dir / f'{config.experiment_name}_config.yaml'
    config.save_yaml(config_path)
    logger.info(f"Config saved to {config_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    # Print test metrics
    logger.info("\nTest Set Metrics:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")


if __name__ == '__main__':
    main()
