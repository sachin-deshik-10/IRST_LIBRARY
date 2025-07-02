# Quick Start Guide

This guide will help you get started with the IRST Library quickly.

## Installation

### From PyPI (Recommended)

```bash
pip install irst-library
```

### From Source

```bash
git clone https://github.com/IRST-Research/irst-library.git
cd irst-library
pip install -e ".[dev]"
```

## Basic Usage

### 1. Simple Detection

```python
from irst_library import IRSTDetector

# Load pretrained model
detector = IRSTDetector.from_pretrained("mshnet")

# Detect targets in an image
results = detector.detect("path/to/infrared_image.png")
print(f"Found {len(results['boxes'])} targets")

# Visualize results
detector.visualize("path/to/infrared_image.png", results, save_path="detection.png")
```

### 2. Batch Processing

```python
import glob
from irst_library import IRSTDetector

detector = IRSTDetector.from_pretrained("mshnet")

# Process multiple images
image_paths = glob.glob("data/images/*.png")
results = detector.detect_batch(image_paths, batch_size=8)

for i, result in enumerate(results):
    print(f"Image {i}: {len(result['boxes'])} targets detected")
```

### 3. Training a Custom Model

```python
from irst_library.trainers import IRSTTrainer
from irst_library.models import MSHNet
from irst_library.datasets import SIRSTDataset

# Prepare data
train_dataset = SIRSTDataset(root="./data/SIRST", split="train")
val_dataset = SIRSTDataset(root="./data/SIRST", split="val")

# Initialize model
model = MSHNet(backbone="resnet18", num_classes=1)

# Configure trainer
trainer = IRSTTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config_path="configs/mshnet_sirst.yaml"
)

# Start training
trainer.fit()
```

## Command Line Interface

### Training

```bash
# Train with config file
irst-train --config configs/mshnet_sirst.yaml --data-root ./data/SIRST

# Train with specific GPU
irst-train --config configs/mshnet_sirst.yaml --gpu 0
```

### Evaluation

```bash
# Evaluate trained model
irst-eval --model checkpoints/best_model.ckpt --dataset sirst

# Save prediction visualizations
irst-eval --model checkpoints/best_model.ckpt --dataset sirst --save-preds
```

### Benchmarking

```bash
# Compare multiple models
irst-benchmark --models mshnet,isnet,dnanet --datasets sirst,nudt-sirst

# Generate HTML report
irst-benchmark --models mshnet,isnet --datasets sirst --output report.html --format html
```

### Demo

```bash
# Launch interactive demo
irst-demo --model mshnet --port 8080
```

## Working with Different Datasets

### SIRST Dataset

```python
from irst_library.datasets import SIRSTDataset

# Standard usage
dataset = SIRSTDataset(root="./data/SIRST", split="train")

# With transforms
import albumentations as A
transforms = A.Compose([
    A.Resize(512, 512),
    A.RandomHorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485], std=[0.229]),
])

dataset = SIRSTDataset(
    root="./data/SIRST", 
    split="train",
    transform=transforms
)
```

### Custom Dataset

```python
from irst_library.datasets import BaseDataset
from irst_library.core.registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)
        # Your dataset initialization logic
        
    def __len__(self):
        # Return dataset size
        pass
        
    def __getitem__(self, idx):
        # Return {"image": image, "mask": mask, "meta": metadata}
        pass
```

## Model Configuration

### Using Configuration Files

```yaml
# config.yaml
model:
  type: MSHNet
  backbone: resnet18
  num_classes: 1
  
dataset:
  type: SIRSTDataset
  root: ./data/SIRST
  batch_size: 8
  
training:
  learning_rate: 0.001
  num_epochs: 100
  optimizer:
    type: Adam
    weight_decay: 1e-4
```

### Programmatic Configuration

```python
from irst_library.models import MSHNet
from irst_library.trainers import IRSTTrainer

# Configure model
model = MSHNet(
    backbone="resnet50",  # Use ResNet-50 backbone
    num_classes=1,
    pretrained=True
)

# Configure training
trainer = IRSTTrainer(
    model=model,
    learning_rate=0.0005,
    batch_size=16,
    max_epochs=50,
    accelerator="gpu",
    devices=2,  # Multi-GPU training
)
```

## Deployment

### Export to ONNX

```bash
# Export model to ONNX format
irst export --model checkpoints/best_model.ckpt --output model.onnx --format onnx
```

### Docker Deployment

```bash
# Build Docker image
docker build -t irst-library .

# Run inference server
docker run -p 8080:8080 --gpus all irst-library

# Run with docker-compose
docker-compose up irst-api
```

### Model Serving

```python
from irst_library.deployment.serving import IRSTServer

# Start serving API
server = IRSTServer(
    model_path="models/mshnet.onnx",
    host="0.0.0.0",
    port=8080
)
server.start()
```

## Performance Optimization

### Mixed Precision Training

```python
trainer = IRSTTrainer(
    model=model,
    precision=16,  # Use half precision
    accelerator="gpu"
)
```

### Model Optimization

```python
from irst_library.deployment import ModelOptimizer

optimizer = ModelOptimizer(model_path="model.onnx")

# Optimize for inference
optimized_model = optimizer.optimize(
    optimization_level="all",
    target_device="gpu"
)
```

## Next Steps

- Check out the [Model Zoo](models.md) for available pretrained models
- Learn about [Dataset Preparation](datasets.md)
- Read the [Training Guide](training.md) for advanced training techniques
- Explore [Deployment Options](deployment.md) for production use

## Getting Help

- 📖 [Full Documentation](https://irst-library.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/IRST-Research/irst-library/discussions)
- 🐛 [Issue Tracker](https://github.com/IRST-Research/irst-library/issues)
- 📧 Email: <contact@irst-lib.org>
