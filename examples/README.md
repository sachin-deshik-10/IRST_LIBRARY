# IRST Library Examples

This directory contains example scripts demonstrating how to use the IRST Library for infrared small target detection.

## Quick Start

### 1. Training a Model

Train a model using a configuration file:

```bash
python examples/train_model.py --config configs/experiments/serank_sirst.yaml
```

Or train with command line arguments:

```bash
python examples/train_model.py --model serank --dataset sirst --epochs 100 --batch-size 16 --lr 0.001
```

### 2. Running Inference

Single image inference:

```bash
python examples/inference.py --model-path checkpoints/serank_best.pth --image-path test_image.png --output-dir results/
```

Batch inference on directory:

```bash
python examples/inference.py --model-path checkpoints/serank_best.pth --input-dir images/ --output-dir results/
```

### 3. Model Evaluation

Evaluate on test dataset:

```bash
python examples/inference.py --model-path checkpoints/serank_best.pth --dataset-config configs/experiments/serank_sirst.yaml --evaluate --output-dir evaluation/
```

## Available Models

- **SERANKNet** (`serank`): Search and ranking network with attention mechanisms
- **ACMNet** (`acm`): Asymmetric contextual modulation network
- **MSHNet** (`mshnet`): Multi-scale feature extraction network
- **Simple U-Net** (`simple_unet`): Basic U-Net architecture

## Available Datasets

- **SIRST** (`sirst`): Single-frame infrared small-target dataset
- **NUDT-SIRST** (`nudt_sirst`): NUDT single-frame infrared small-target dataset
- **IRSTD-1K** (`irstd1k`): Large-scale infrared small target dataset
- **NUAA-SIRST** (`nuaa_sirst`): NUAA single-frame infrared small-target dataset

## Configuration Files

Configuration files are located in `configs/experiments/`:

- `serank_sirst.yaml`: SERANKNet on SIRST dataset
- `acm_nuaa_sirst.yaml`: ACMNet on NUAA-SIRST dataset
- `mshnet_sirst.yaml`: MSHNet on SIRST dataset

### Configuration Structure

```yaml
# Experiment info
name: "experiment_name"
description: "Experiment description"

# Model configuration
model:
  name: "model_name"
  # model-specific parameters

# Dataset configuration
dataset:
  name: "dataset_name"
  root_dir: "path/to/dataset"
  # dataset-specific parameters

# Training configuration
training:
  batch_size: 16
  num_epochs: 100
  optimizer:
    name: "adamw"
    lr: 0.001
  loss:
    name: "irst"
  # other training parameters

# Callbacks and other configurations
```

## Training Process

The training script supports:

- **Multiple optimizers**: Adam, AdamW, SGD, RMSprop
- **Learning rate schedulers**: Step, Cosine, Plateau, Exponential
- **Loss functions**: BCE, Dice, IoU, Focal, Tversky, Combined IRST loss
- **Callbacks**: Early stopping, model checkpointing, LR reduction, logging
- **Metrics**: Precision, Recall, F1, IoU, AUC-ROC, AUC-PR
- **Data augmentation**: Rotation, scaling, flipping, noise, brightness/contrast

## Evaluation Metrics

The library computes comprehensive evaluation metrics:

### Pixel-level Metrics

- Precision, Recall, F1-score
- Intersection over Union (IoU)
- Pixel accuracy

### Object-level Metrics

- Detection precision and recall
- True/false positive counts
- Target localization accuracy

### Threshold Analysis

- ROC curves and AUC-ROC
- Precision-Recall curves and AUC-PR
- Optimal threshold selection

## Example Workflows

### 1. Train SERANKNet on SIRST

```bash
# Download and prepare SIRST dataset
mkdir -p data/sirst

# Train model
python examples/train_model.py \
    --config configs/experiments/serank_sirst.yaml \
    --plot

# Evaluate trained model
python examples/inference.py \
    --model-path checkpoints/serank_sirst_best.pth \
    --dataset-config configs/experiments/serank_sirst.yaml \
    --evaluate \
    --output-dir evaluation/serank_sirst/
```

### 2. Fine-tune Pretrained Model

```bash
# Resume training from checkpoint
python examples/train_model.py \
    --config configs/experiments/serank_sirst.yaml \
    --resume checkpoints/serank_epoch_50.pth \
    --epochs 100
```

### 3. Cross-dataset Evaluation

```bash
# Train on SIRST
python examples/train_model.py --config configs/experiments/serank_sirst.yaml

# Evaluate on NUAA-SIRST
python examples/inference.py \
    --model-path checkpoints/serank_sirst_best.pth \
    --dataset-config configs/experiments/acm_nuaa_sirst.yaml \
    --evaluate
```

## Advanced Usage

### Custom Model Integration

```python
from irst_library.core.registry import MODEL_REGISTRY
from irst_library.core.base import BaseModel

@MODEL_REGISTRY.register("my_model")
class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Define your model architecture
    
    def forward(self, x):
        # Forward pass implementation
        return x
```

### Custom Dataset Integration

```python
from irst_library.core.registry import DATASET_REGISTRY
from irst_library.core.base import BaseDataset

@DATASET_REGISTRY.register("my_dataset")
class MyDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__()
        # Dataset initialization
    
    def __getitem__(self, idx):
        # Return sample dictionary
        return {"image": image, "mask": mask}
```

### Programmatic Training

```python
from irst_library.training import IRSTTrainer
from irst_library.models import get_model
from irst_library.datasets import get_dataset

# Create model and datasets
model = get_model("serank", in_channels=1, num_classes=1)
train_dataset = get_dataset("sirst", split="train")
val_dataset = get_dataset("sirst", split="val")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Create trainer
trainer = IRSTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer="adamw",
    loss_function="irst"
)

# Train
history = trainer.fit(num_epochs=100)

# Evaluate
results = trainer.evaluate(test_loader)
```

## Output Files

Training and evaluation generate several output files:

```
outputs/
├── checkpoints/
│   ├── model_epoch_best.pth
│   └── model_final.pth
├── logs/
│   ├── training.csv
│   ├── tensorboard/
│   └── training.log
├── predictions/
│   ├── image1_prediction.png
│   └── image1_overlay.png
└── evaluation/
    ├── evaluation_results.json
    ├── confusion_matrix.png
    └── roc_curves.png
```

## Performance Tips

1. **Use mixed precision training** for faster training and lower memory usage
2. **Cache datasets in memory** for small datasets to speed up data loading
3. **Use appropriate batch sizes** based on GPU memory
4. **Enable data augmentation** to improve model generalization
5. **Monitor training with TensorBoard** for better debugging

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Dataset not found**: Check data paths in configuration files
3. **Model loading errors**: Ensure checkpoint compatibility
4. **Slow training**: Increase num_workers or use data caching

### Debug Mode

Enable debug logging for more detailed information:

```bash
python examples/train_model.py --config config.yaml --log-level DEBUG
```

## Support

For questions and issues:

1. Check the main documentation
2. Review configuration examples
3. Search existing GitHub issues
4. Create a new issue with detailed information
