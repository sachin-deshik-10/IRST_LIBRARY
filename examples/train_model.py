#!/usr/bin/env python3
"""
Training example for IRST Library.

This script demonstrates how to train an infrared small target detection model
using the IRST Library with various models and datasets.

Usage:
    python examples/train_model.py --config configs/experiments/serank_sirst.yaml
    python examples/train_model.py --model serank --dataset sirst --epochs 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import yaml

# Add the library to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irst_library.core.registry import MODEL_REGISTRY, DATASET_REGISTRY
from irst_library.training import (
    IRSTTrainer, create_trainer,
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger, TensorBoardLogger
)
from irst_library.utils.visualization import plot_training_history


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """Create train/val/test datasets"""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    # Get dataset class
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create transforms
    image_size = tuple(dataset_config['image_size'])
    
    # Training transforms (with augmentation)
    train_transforms = dataset_class.get_default_transforms(
        image_size=image_size,
        is_training=True
    )
    
    # Validation transforms (no augmentation)
    val_transforms = dataset_class.get_default_transforms(
        image_size=image_size,
        is_training=False
    )
    
    # Create datasets
    common_kwargs = {
        'root_dir': dataset_config['root_dir'],
        'cache_in_memory': dataset_config.get('cache_in_memory', False),
        'return_paths': False
    }
    
    train_dataset = dataset_class(
        split=dataset_config['train_split'],
        transform=train_transforms,
        **common_kwargs
    )
    
    val_dataset = dataset_class(
        split=dataset_config['val_split'],
        transform=val_transforms,
        **common_kwargs
    )
    
    test_dataset = dataset_class(
        split=dataset_config['test_split'],
        transform=val_transforms,
        **common_kwargs
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, config: dict):
    """Create data loaders"""
    dataset_config = config['dataset']
    training_config = config['training']
    
    # Common data loader kwargs
    common_kwargs = {
        'num_workers': dataset_config.get('num_workers', 4),
        'pin_memory': dataset_config.get('pin_memory', True),
        'persistent_workers': True if dataset_config.get('num_workers', 4) > 0 else False
    }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        drop_last=True,
        **common_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        drop_last=False,
        **common_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        drop_last=False,
        **common_kwargs
    )
    
    return train_loader, val_loader, test_loader


def create_model(config: dict):
    """Create model from configuration"""
    model_config = config['model']
    model_name = model_config['name']
    
    # Get model class
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Remove 'name' from config and pass rest as kwargs
    model_kwargs = {k: v for k, v in model_config.items() if k != 'name'}
    
    model = model_class(**model_kwargs)
    return model


def create_optimizer(model, config: dict):
    """Create optimizer from configuration"""
    opt_config = config['training']['optimizer']
    opt_name = opt_config['name'].lower()
    
    # Remove 'name' from config
    opt_kwargs = {k: v for k, v in opt_config.items() if k != 'name'}
    
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), **opt_kwargs)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **opt_kwargs)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), **opt_kwargs)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), **opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler"""
    if 'scheduler' not in config['training']:
        return None
    
    sched_config = config['training']['scheduler']
    sched_name = sched_config['name'].lower()
    
    # Remove 'name' from config
    sched_kwargs = {k: v for k, v in sched_config.items() if k != 'name'}
    
    if sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_kwargs)
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_kwargs)
    elif sched_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_kwargs)
    elif sched_name == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **sched_kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")
    
    return scheduler


def create_callbacks(config: dict):
    """Create training callbacks"""
    callbacks = []
    callback_config = config.get('callbacks', {})
    
    # Early stopping
    if callback_config.get('early_stopping', {}).get('enabled', False):
        es_config = callback_config['early_stopping']
        callbacks.append(EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            mode=es_config.get('mode', 'min'),
            restore_best_weights=es_config.get('restore_best_weights', True),
            verbose=True
        ))
    
    # Model checkpoint
    if callback_config.get('model_checkpoint', {}).get('enabled', False):
        cp_config = callback_config['model_checkpoint']
        
        # Create checkpoint directory
        filepath = cp_config['filepath'].format(name=config['name'])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        callbacks.append(ModelCheckpoint(
            filepath=filepath,
            monitor=cp_config.get('monitor', 'val_loss'),
            mode=cp_config.get('mode', 'min'),
            save_best_only=cp_config.get('save_best_only', True),
            verbose=True
        ))
    
    # Reduce LR on plateau
    if callback_config.get('reduce_lr', {}).get('enabled', False):
        lr_config = callback_config['reduce_lr']
        callbacks.append(ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 5),
            mode=lr_config.get('mode', 'min'),
            verbose=True
        ))
    
    # CSV logger
    if callback_config.get('csv_logger', {}).get('enabled', False):
        csv_config = callback_config['csv_logger']
        filename = csv_config['filename'].format(name=config['name'])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        callbacks.append(CSVLogger(filename))
    
    # TensorBoard logger
    if callback_config.get('tensorboard', {}).get('enabled', False):
        tb_config = callback_config['tensorboard']
        log_dir = tb_config['log_dir'].format(name=config['name'])
        callbacks.append(TensorBoardLogger(log_dir))
    
    return callbacks


def main(args):
    """Main training function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Create config from command line arguments
        config = {
            'name': f"{args.model}_{args.dataset}",
            'model': {'name': args.model},
            'dataset': {'name': args.dataset, 'root_dir': args.data_dir},
            'training': {
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'optimizer': {'name': 'adam', 'lr': args.lr}
            },
            'system': {'device': args.device, 'seed': args.seed}
        }
    
    # Set random seed
    if 'system' in config and 'seed' in config['system']:
        torch.manual_seed(config['system']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['system']['seed'])
    
    logger.info(f"Starting training: {config['name']}")
    logger.info(f"Configuration: {config}")
    
    # Create datasets and data loaders
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Validation: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create trainer
    trainer = IRSTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=config['training'].get('loss', {}).get('name', 'irst'),
        device=config.get('system', {}).get('device', 'auto'),
        callbacks=callbacks
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.fit(
        num_epochs=config['training']['num_epochs'],
        verbose=1
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    logger.info(f"Test results: {test_results}")
    
    # Save final model
    output_dir = Path(config.get('paths', {}).get('output_dir', 'outputs'))
    output_dir.mkdir(exist_ok=True)
    
    final_model_path = output_dir / f"{config['name']}_final.pth"
    trainer.save_checkpoint(final_model_path, test_results=test_results)
    
    # Plot training history
    if args.plot:
        plot_path = output_dir / f"{config['name']}_history.png"
        plot_training_history(history, save_path=plot_path)
        logger.info(f"Training history plot saved: {plot_path}")
    
    logger.info("Training completed successfully!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train IRST detection model")
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    # Model and dataset
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="serank",
        choices=["mshnet", "serank", "acm", "simple_unet"],
        help="Model architecture"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="sirst",
        choices=["sirst", "nudt_sirst", "irstd1k", "nuaa_sirst"],
        help="Dataset to use"
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to dataset directory"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Visualization
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training history"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
