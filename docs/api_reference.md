# API Reference

This document provides a comprehensive reference for the IRST Library API.

## Table of Contents

- [Core API](#core-api)
- [Models](#models)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Utilities](#utilities)
- [CLI Commands](#cli-commands)

## Core API

### IRSTDetector

The main interface for infrared small target detection.

```python
from irst_library import IRSTDetector
```

#### Methods

##### `from_pretrained(model_name: str, **kwargs) -> IRSTDetector`

Load a pretrained model.

**Parameters:**

- `model_name` (str): Name of the pretrained model
- `**kwargs`: Additional arguments for model configuration

**Returns:**

- `IRSTDetector`: Configured detector instance

**Example:**

```python
detector = IRSTDetector.from_pretrained("serank_sirst")
```

##### `detect(image: Union[str, np.ndarray, torch.Tensor]) -> DetectionResult`

Perform detection on an input image.

**Parameters:**

- `image`: Input image (file path, numpy array, or torch tensor)

**Returns:**

- `DetectionResult`: Detection results with masks and scores

**Example:**

```python
results = detector.detect("infrared_image.png")
print(f"Detected {len(results.targets)} targets")
```

## Models

### Model Registry

Access available models through the registry system.

```python
from irst_library.models import get_model, list_models
```

#### Functions

##### `get_model(name: str, **kwargs) -> torch.nn.Module`

Create a model instance.

**Parameters:**

- `name` (str): Model name ("serank", "acm", "mshnet", "unet")
- `**kwargs`: Model-specific parameters

**Example:**

```python
model = get_model("serank", in_channels=1, num_classes=1)
```

##### `list_models() -> List[str]`

List all available models.

**Returns:**

- `List[str]`: Available model names

### SERANKNet

Search and ranking network with attention mechanisms.

```python
from irst_library.models.single_frame import SERANKNet
```

#### Parameters

- `in_channels` (int): Input channels (default: 1)
- `num_classes` (int): Number of output classes (default: 1)
- `base_channels` (int): Base channel count (default: 64)

#### Example

```python
model = SERANKNet(in_channels=1, num_classes=1, base_channels=64)
```

### ACMNet

Asymmetric contextual modulation network.

```python
from irst_library.models.single_frame import ACMNet
```

#### Parameters

- `in_channels` (int): Input channels (default: 1)
- `num_classes` (int): Number of output classes (default: 1)

## Datasets

### Dataset Registry

```python
from irst_library.datasets import get_dataset, list_datasets
```

#### Functions

##### `get_dataset(name: str, **kwargs) -> torch.utils.data.Dataset`

Create a dataset instance.

**Parameters:**

- `name` (str): Dataset name
- `**kwargs`: Dataset-specific parameters

**Example:**

```python
dataset = get_dataset("sirst", root_dir="data/sirst", split="train")
```

### SIRST Dataset

```python
from irst_library.datasets import SIRSTDataset
```

#### Parameters

- `root_dir` (str): Path to dataset root
- `split` (str): "train", "val", or "test"
- `transform` (callable): Data transformation pipeline
- `image_size` (tuple): Target image size

#### Example

```python
from irst_library.datasets import SIRSTDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = SIRSTDataset(
    root_dir="data/sirst",
    split="train",
    transform=transform,
    image_size=(256, 256)
)
```

## Training

### IRSTTrainer

Main training interface.

```python
from irst_library.training import IRSTTrainer
```

#### Parameters

- `model` (torch.nn.Module): Model to train
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader
- `optimizer` (torch.optim.Optimizer): Optimizer
- `loss_fn` (torch.nn.Module): Loss function
- `device` (str): Training device

#### Methods

##### `fit(num_epochs: int, **kwargs) -> TrainingHistory`

Train the model.

**Parameters:**

- `num_epochs` (int): Number of training epochs
- `**kwargs`: Additional training parameters

**Returns:**

- `TrainingHistory`: Training metrics and history

#### Example

```python
from irst_library.training import IRSTTrainer
from irst_library.training.losses import IRSTLoss

trainer = IRSTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
    loss_fn=IRSTLoss(),
    device="cuda"
)

history = trainer.fit(num_epochs=100)
```

### Loss Functions

#### IRSTLoss

Combined loss function for ISTD tasks.

```python
from irst_library.training.losses import IRSTLoss
```

**Parameters:**

- `dice_weight` (float): Weight for Dice loss (default: 1.0)
- `focal_weight` (float): Weight for Focal loss (default: 1.0)
- `iou_weight` (float): Weight for IoU loss (default: 1.0)

#### DiceLoss

Dice coefficient loss for segmentation.

```python
from irst_library.training.losses import DiceLoss
```

**Parameters:**

- `smooth` (float): Smoothing factor (default: 1e-6)

## Evaluation

### Metrics

```python
from irst_library.training.metrics import (
    PixelMetrics,
    ObjectMetrics,
    ThresholdAnalysis
)
```

#### PixelMetrics

Pixel-level evaluation metrics.

**Methods:**

- `precision()`: Pixel-level precision
- `recall()`: Pixel-level recall
- `f1_score()`: F1-score
- `iou()`: Intersection over Union

#### ObjectMetrics

Object-level evaluation metrics.

**Methods:**

- `detection_rate()`: Target detection rate
- `false_alarm_rate()`: False alarm rate
- `localization_error()`: Average localization error

#### Example

```python
from irst_library.training.metrics import PixelMetrics

metrics = PixelMetrics()
metrics.update(predictions, targets)

print(f"IoU: {metrics.iou():.4f}")
print(f"F1: {metrics.f1_score():.4f}")
```

## Utilities

### Visualization

```python
from irst_library.utils.visualization import (
    plot_predictions,
    plot_training_history,
    create_overlay
)
```

#### Functions

##### `plot_predictions(image, prediction, target=None, **kwargs)`

Visualize model predictions.

**Parameters:**

- `image`: Input image
- `prediction`: Model prediction
- `target`: Ground truth (optional)
- `**kwargs`: Plotting options

##### `create_overlay(image, mask, alpha=0.5, color='red')`

Create prediction overlay on image.

**Parameters:**

- `image`: Base image
- `mask`: Binary mask
- `alpha`: Overlay transparency
- `color`: Overlay color

### Image Processing

```python
from irst_library.utils.image import (
    preprocess_image,
    postprocess_prediction,
    normalize_image
)
```

## CLI Commands

### Training

```bash
# Train a model
irst-train --config configs/experiments/serank_sirst.yaml

# Options:
#   --config: Path to configuration file
#   --model: Model name
#   --dataset: Dataset name
#   --epochs: Number of epochs
#   --batch-size: Batch size
#   --lr: Learning rate
```

### Inference

```bash
# Run inference
irst-detect --model-path checkpoints/best.pth --image-path image.png

# Options:
#   --model-path: Path to trained model
#   --image-path: Path to input image
#   --output-path: Output directory
#   --threshold: Detection threshold
#   --device: Computing device
```

### Evaluation

```bash
# Evaluate model
irst-evaluate --model-path checkpoints/best.pth --dataset-config config.yaml

# Options:
#   --model-path: Path to trained model
#   --dataset-config: Dataset configuration
#   --output-dir: Results directory
#   --metrics: Evaluation metrics
```

### Model Export

```bash
# Export model
irst-export --model-path checkpoints/best.pth --format onnx

# Options:
#   --model-path: Path to trained model
#   --format: Export format (onnx, tensorrt)
#   --output-path: Output file path
#   --input-shape: Model input shape
```

## Configuration

### Model Configuration

```yaml
model:
  name: "serank"
  in_channels: 1
  num_classes: 1
  base_channels: 64
```

### Dataset Configuration

```yaml
dataset:
  name: "sirst"
  root_dir: "data/sirst"
  image_size: [256, 256]
  augmentation:
    rotation: 15
    scale: [0.8, 1.2]
    noise: 0.1
```

### Training Configuration

```yaml
training:
  batch_size: 16
  num_epochs: 100
  optimizer:
    name: "adamw"
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    name: "cosine"
    warmup_epochs: 10
  loss:
    name: "irst"
    dice_weight: 1.0
    focal_weight: 1.0
    iou_weight: 1.0
```

## Error Handling

The library provides comprehensive error handling with custom exceptions:

```python
from irst_library.core.exceptions import (
    ModelNotFoundError,
    DatasetNotFoundError,
    ConfigurationError,
    InferenceError
)

try:
    model = get_model("invalid_model")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
```

## Type Hints

The library provides full type hint support:

```python
from typing import Optional, Union, List, Dict, Any
import torch
import numpy as np

def detect_targets(
    image: Union[str, np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    device: Optional[str] = None
) -> Dict[str, Any]:
    ...
```

---

For more detailed examples and tutorials, see the [examples](../examples/) directory.
