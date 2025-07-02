#!/usr/bin/env python3
"""
Inference and evaluation example for IRST Library.

This script demonstrates how to load a trained model and perform inference
on images or evaluate on a test dataset.

Usage:
    # Inference on single image
    python examples/inference.py --model-path checkpoints/serank_best.pth --image-path test_image.png

    # Evaluation on test dataset
    python examples/inference.py --model-path checkpoints/serank_best.pth --dataset-config configs/experiments/serank_sirst.yaml --evaluate

    # Batch inference on directory
    python examples/inference.py --model-path checkpoints/serank_best.pth --input-dir images/ --output-dir results/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# Add the library to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irst_library.core.registry import MODEL_REGISTRY, DATASET_REGISTRY
from irst_library.core.detector import IRSTDetector
from irst_library.training import MetricsCalculator
from irst_library.utils.image import load_image, normalize_image, denormalize_image
from irst_library.utils.visualization import visualize_detection, create_overlay


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load model from checkpoint"""
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model info from checkpoint if available
    model_config = checkpoint.get('model_config', {})
    model_name = model_config.get('name', 'serank')  # Default to serank
    
    # Get model class
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model
    model_kwargs = {k: v for k, v in model_config.items() if k != 'name'}
    model = model_class(**model_kwargs)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path: str, target_size: tuple = (256, 256)):
    """Preprocess single image for inference"""
    # Load image
    image = load_image(image_path, grayscale=True)
    original_size = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    # Normalize
    image_tensor = normalize_image(image_tensor)
    
    return image_tensor, original_size


def postprocess_prediction(prediction: torch.Tensor, original_size: tuple, threshold: float = 0.5):
    """Postprocess prediction tensor"""
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Resize to original size
    pred_resized = cv2.resize(pred_np, (original_size[1], original_size[0]))
    
    # Apply threshold
    binary_mask = (pred_resized > threshold).astype(np.uint8) * 255
    
    return pred_resized, binary_mask


def single_image_inference(model, image_path: str, device, threshold: float = 0.5, save_path: str = None):
    """Perform inference on single image"""
    logger = logging.getLogger(__name__)
    
    # Preprocess image
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Postprocess
    pred_map, binary_mask = postprocess_prediction(prediction, original_size, threshold)
    
    # Load original image for visualization
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create visualization
    overlay = create_overlay(original_image, binary_mask)
    
    # Save results if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save prediction map
        cv2.imwrite(str(save_path / "prediction.png"), (pred_map * 255).astype(np.uint8))
        
        # Save binary mask
        cv2.imwrite(str(save_path / "binary_mask.png"), binary_mask)
        
        # Save overlay
        cv2.imwrite(str(save_path / "overlay.png"), overlay)
        
        logger.info(f"Results saved to: {save_path}")
    
    return {
        'prediction': pred_map,
        'binary_mask': binary_mask,
        'overlay': overlay,
        'original_image': original_image
    }


def batch_inference(model, input_dir: str, output_dir: str, device, threshold: float = 0.5):
    """Perform batch inference on directory of images"""
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Create output directory for this image
            image_output_dir = output_path / image_file.stem
            
            # Perform inference
            results = single_image_inference(
                model, str(image_file), device, threshold, str(image_output_dir)
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            continue
    
    logger.info(f"Batch inference completed. Results saved to: {output_dir}")


def evaluate_on_dataset(model, config: dict, device, threshold: float = 0.5):
    """Evaluate model on test dataset"""
    logger = logging.getLogger(__name__)
    
    # Create test dataset
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create transforms (no augmentation for evaluation)
    image_size = tuple(dataset_config['image_size'])
    transforms = dataset_class.get_default_transforms(image_size, is_training=False)
    
    # Create test dataset
    test_dataset = dataset_class(
        root_dir=dataset_config['root_dir'],
        split=dataset_config.get('test_split', 'test'),
        transform=transforms,
        return_paths=True
    )
    
    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for evaluation
        shuffle=False,
        num_workers=0
    )
    
    # Initialize metrics
    metrics_calculator = MetricsCalculator(
        thresholds=[threshold],
        min_object_area=config.get('evaluation', {}).get('min_object_area', 10),
        distance_threshold=config.get('evaluation', {}).get('distance_threshold', 3.0)
    )
    
    # Evaluate
    logger.info(f"Evaluating on {len(test_dataset)} images...")
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move to device
            images = batch['image'].to(device)
            target_masks = batch['mask'].to(device)
            
            # Forward pass
            pred_masks = model(images)
            
            # Update metrics
            metrics_calculator.update(pred_masks, target_masks)
            
            # Store for later analysis
            predictions.append(pred_masks.cpu().numpy())
            targets.append(target_masks.cpu().numpy())
    
    # Compute final metrics
    results = metrics_calculator.compute()
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("==================")
    
    summary_metrics = results['summary']
    for metric_name, value in summary_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Detailed metrics
    logger.info("\nDetailed Metrics:")
    logger.info("-----------------")
    
    # Binary metrics
    binary_metrics = results[f'binary_t{threshold}']
    logger.info(f"Threshold {threshold}:")
    logger.info(f"  Precision: {binary_metrics['precision']:.4f}")
    logger.info(f"  Recall: {binary_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {binary_metrics['f1']:.4f}")
    logger.info(f"  IoU: {binary_metrics['iou']:.4f}")
    logger.info(f"  AUC-ROC: {binary_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {binary_metrics['auc_pr']:.4f}")
    
    # Object-level metrics
    object_metrics = results['object']
    logger.info(f"\nObject-level Metrics:")
    logger.info(f"  Precision: {object_metrics['object_precision']:.4f}")
    logger.info(f"  Recall: {object_metrics['object_recall']:.4f}")
    logger.info(f"  F1-Score: {object_metrics['object_f1']:.4f}")
    logger.info(f"  True Detections: {object_metrics['true_detections']}")
    logger.info(f"  False Detections: {object_metrics['false_detections']}")
    logger.info(f"  Missed Targets: {object_metrics['missed_targets']}")
    
    return results


def create_detector_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Create IRSTDetector from checkpoint"""
    model, device = load_model_from_checkpoint(checkpoint_path, device)
    
    detector = IRSTDetector(model=model, device=device)
    return detector


def main(args):
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model, device = load_model_from_checkpoint(args.model_path, args.device)
    logger.info(f"Model loaded successfully on device: {device}")
    
    # Single image inference
    if args.image_path:
        logger.info(f"Processing single image: {args.image_path}")
        results = single_image_inference(
            model, args.image_path, device, args.threshold, args.output_dir
        )
        
        # Display results if requested
        if args.display:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(results['original_image'], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(results['prediction'], cmap='hot')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(results['overlay'])
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # Batch inference
    elif args.input_dir:
        logger.info(f"Processing batch inference: {args.input_dir} -> {args.output_dir}")
        batch_inference(model, args.input_dir, args.output_dir, device, args.threshold)
    
    # Dataset evaluation
    elif args.evaluate and args.dataset_config:
        logger.info(f"Evaluating on dataset: {args.dataset_config}")
        config = load_config(args.dataset_config)
        results = evaluate_on_dataset(model, config, device, args.threshold)
        
        # Save evaluation results
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            import json
            with open(output_path / "evaluation_results.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                           for k, v in value.items()}
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to: {output_path}")
    
    else:
        logger.error("Please specify either --image-path, --input-dir, or --evaluate with --dataset-config")
        return 1
    
    logger.info("Inference completed successfully!")
    return 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="IRST model inference and evaluation")
    
    # Model
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Input options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image-path", "-i",
        type=str,
        help="Path to input image for single inference"
    )
    
    group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing input images for batch inference"
    )
    
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on test dataset"
    )
    
    # Dataset configuration (required for evaluation)
    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Path to dataset configuration YAML file (required for evaluation)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="inference_results",
        help="Output directory for results"
    )
    
    # Inference parameters
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection threshold"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    
    # Visualization
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display results (for single image inference)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
