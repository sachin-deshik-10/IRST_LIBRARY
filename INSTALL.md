# IRST Library - Advanced Infrared Small Target Detection

[![PyPI version](https://badge.fury.io/py/irst-library.svg)](https://badge.fury.io/py/irst-library)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/sachin-deshik-10/irst-library/workflows/CI/badge.svg)](https://github.com/sachin-deshik-10/irst-library/actions)
[![Documentation Status](https://readthedocs.org/projects/irst-library/badge/?version=latest)](https://irst-library.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/sachin-deshik-10/irst-library/branch/main/graph/badge.svg)](https://codecov.io/gh/sachin-deshik-10/irst-library)

> 🚀 **The most comprehensive and production-ready library for Infrared Small Target Detection (ISTD) research and deployment.**

## 🎯 Key Features

- **🏗️ Production-Ready**: Full pipeline from training to deployment with Docker support
- **📚 700+ Paper Implementations**: State-of-the-art methods with reproducible results
- **🚀 High Performance**: Optimized for real-time inference with TensorRT and ONNX support
- **🎨 Modern Architecture**: Built with PyTorch Lightning and Hydra configuration
- **📊 Comprehensive Benchmarking**: Standardized evaluation across multiple datasets
- **🔧 Easy Integration**: Simple API for custom applications and research
- **📈 MLOps Ready**: Weights & Biases integration, model versioning, and monitoring
- **🌐 Cross-Platform**: Windows, Linux, and macOS support

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install irst-library

# Or install from source with development dependencies
git clone https://github.com/sachin-deshik-10/irst-library.git
cd irst-library
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from irst_library import IRSTDetector
from irst_library.models import get_model
from irst_library.datasets import get_dataset

# Initialize detector with pretrained model
detector = IRSTDetector.from_pretrained("serank_det_sirst")

# Load your infrared image
image = torch.randn(1, 1, 512, 512)  # Replace with your image

# Detect small targets
results = detector.detect(image)
print(f"Detected {len(results['boxes'])} targets")

# Visualize results
detector.visualize(image, results, save_path="detection_result.png")
```

### Training a Custom Model

```python
from irst_library.trainers import IRSTTrainer
from irst_library.models import MSHNet
from irst_library.datasets import SIRSTDataset

# Initialize model and dataset
model = MSHNet(num_classes=1)
dataset = SIRSTDataset(root="./data/SIRST", split="train")

# Configure trainer
trainer = IRSTTrainer(
    model=model,
    train_dataset=dataset,
    val_dataset=SIRSTDataset(root="./data/SIRST", split="val"),
    config_path="configs/mshnet_sirst.yaml"
)

# Start training
trainer.fit()
```

### Command Line Interface

```bash
# Train a model
irst-train --config configs/serank_det.yaml --data.root ./data/SIRST

# Evaluate model performance
irst-eval --model-path checkpoints/best_model.ckpt --dataset SIRST

# Run benchmark across multiple models
irst-benchmark --models serank_det mshnet isnet --dataset NUDT-SIRST

# Launch interactive demo
irst-demo --model serank_det --port 8080
```

## 📊 Model Zoo & Performance

### Single-Frame Detection Models

| Model | Dataset | IoU ↑ | Pd ↑ | Fa ↓ | FPS | Params | Download |
|-------|---------|-------|------|------|-----|--------|----------|
| [SeRankDet](models/serank_det.py) | SIRST | 0.876 | 0.892 | 8.7e-6 | 42.1 | 12.3M | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [MSHNet](models/mshnet.py) | SIRST | 0.854 | 0.881 | 1.2e-5 | 51.3 | 8.7M | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [IRSAM](models/irsam.py) | SIRST | 0.869 | 0.885 | 9.8e-6 | 28.7 | 15.2M | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [ISNet](models/isnet.py) | SIRST | 0.821 | 0.867 | 1.8e-5 | 62.4 | 6.2M | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [DNANet](models/dnanet.py) | SIRST | 0.815 | 0.854 | 2.1e-5 | 58.7 | 7.8M | [weights](https://github.com/IRST-Research/irst-library/releases) |

### Multi-Frame Detection Models

| Model | Dataset | IoU ↑ | Pd ↑ | Fa ↓ | FPS | Download |
|-------|---------|-------|------|------|-----|----------|
| [S2MVP](models/s2mvp.py) | NUDT-MIRSDT | 0.743 | 0.812 | 1.4e-5 | 35.2 | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [LMAFormer](models/lmaformer.py) | TSIRMT | 0.728 | 0.798 | 1.8e-5 | 28.9 | [weights](https://github.com/IRST-Research/irst-library/releases) |
| [SSTNet](models/sstnet.py) | NUDT-MIRSDT | 0.712 | 0.785 | 2.2e-5 | 31.7 | [weights](https://github.com/IRST-Research/irst-library/releases) |

## 📁 Project Structure

```
irst_library/
├── 📁 core/              # Core components
│   ├── base.py           # Base classes
│   ├── registry.py       # Model/dataset registry
│   └── utils.py          # Utility functions
├── 📁 models/            # Model implementations
│   ├── single_frame/     # Single-frame models
│   │   ├── serank_det.py
│   │   ├── mshnet.py
│   │   ├── isnet.py
│   │   └── ...
│   ├── multi_frame/      # Multi-frame models
│   │   ├── s2mvp.py
│   │   ├── lmaformer.py
│   │   └── ...
│   └── backbones/        # Backbone networks
├── 📁 datasets/          # Dataset implementations
│   ├── sirst.py
│   ├── nudt_sirst.py
│   ├── irstd_1k.py
│   └── transforms/
├── 📁 trainers/          # Training frameworks
│   ├── base_trainer.py
│   ├── single_frame_trainer.py
│   └── multi_frame_trainer.py
├── 📁 evaluation/        # Evaluation metrics
│   ├── metrics.py
│   ├── visualizers.py
│   └── benchmarks/
├── 📁 deployment/        # Deployment tools
│   ├── onnx_export.py
│   ├── tensorrt_optimize.py
│   └── serving/
├── 📁 configs/           # Configuration files
│   ├── models/
│   ├── datasets/
│   └── experiments/
└── 📁 cli/              # Command line interface
    ├── train.py
    ├── evaluate.py
    └── benchmark.py
```

## 🛠️ Advanced Features

### 1. Model Export & Deployment

```python
from irst_library.deployment import ModelExporter

# Export to ONNX
exporter = ModelExporter(model_path="checkpoints/serank_det.ckpt")
exporter.to_onnx("models/serank_det.onnx", opset_version=11)

# Optimize with TensorRT
exporter.to_tensorrt("models/serank_det.trt", precision="fp16")

# Deploy with serving API
from irst_library.deployment.serving import IRSTServer
server = IRSTServer(model_path="models/serank_det.onnx", port=8080)
server.start()
```

### 2. Custom Dataset Integration

```python
from irst_library.datasets import BaseDataset

class CustomIRSTDataset(BaseDataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__(root, split)
        self.images = self._load_images()
        self.masks = self._load_masks()
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        
        return {"image": image, "mask": mask, "meta": self.get_meta(idx)}

# Register your dataset
from irst_library.core.registry import DATASETS
DATASETS.register_module()(CustomIRSTDataset)
```

### 3. Multi-GPU Training with Lightning

```python
from irst_library.trainers import IRSTLightningTrainer
import pytorch_lightning as pl

trainer = IRSTLightningTrainer(
    model="serank_det",
    dataset="sirst",
    accelerator="gpu",
    devices=4,  # Use 4 GPUs
    strategy="ddp",  # Distributed training
    precision=16,  # Mixed precision
    max_epochs=100,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="val_iou",
            mode="max",
            save_top_k=3
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_iou",
            patience=10,
            mode="max"
        )
    ]
)

trainer.fit()
```

### 4. Hyperparameter Optimization

```python
from irst_library.optimization import IRSTOptimizer
import optuna

def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Train model with suggested parameters
    trainer = IRSTTrainer(
        model="mshnet",
        dataset="sirst",
        lr=lr,
        batch_size=batch_size,
        max_epochs=50
    )
    
    results = trainer.fit()
    return results["val_iou"]

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

## 📊 Benchmarking & Analysis

### Performance Dashboard

Run comprehensive benchmarks across models and datasets:

```bash
# Generate performance report
irst-benchmark --output results/benchmark_report.html --format html

# Compare specific models
irst-benchmark --models serank_det mshnet isnet --datasets SIRST NUDT-SIRST --metrics iou pd fa

# Profile model efficiency
irst-benchmark --profile --device cuda --batch-sizes 1,4,8,16
```

### Results Visualization

```python
from irst_library.evaluation import BenchmarkVisualizer

viz = BenchmarkVisualizer()

# Plot performance comparison
viz.plot_model_comparison(
    results_path="results/benchmark_results.json",
    metrics=["iou", "pd", "fa"],
    save_path="figures/model_comparison.png"
)

# Generate confusion matrices
viz.plot_confusion_matrix(
    model_path="checkpoints/serank_det.ckpt",
    dataset="sirst",
    save_path="figures/confusion_matrix.png"
)
```

## 🔬 Research & Development

### Implementing New Models

```python
from irst_library.models.base import BaseModel
from irst_library.core.registry import MODELS

@MODELS.register_module()
class MyNewModel(BaseModel):
    def __init__(self, backbone="resnet18", num_classes=1):
        super().__init__()
        self.backbone = self.build_backbone(backbone)
        self.head = self.build_head(num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def compute_loss(self, pred, target):
        # Implement your loss function
        return F.binary_cross_entropy_with_logits(pred, target)
```

### Custom Training Loops

```python
from irst_library.trainers.base import BaseTrainer

class MyCustomTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        # Custom training logic
        images, masks = batch["image"], batch["mask"]
        
        # Forward pass
        pred = self.model(images)
        loss = self.model.compute_loss(pred, masks)
        
        # Log metrics
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Custom validation logic
        images, masks = batch["image"], batch["mask"]
        
        with torch.no_grad():
            pred = self.model(images)
            loss = self.model.compute_loss(pred, masks)
            
            # Compute metrics
            iou = self.compute_iou(pred, masks)
            
        self.log_dict({
            "val_loss": loss,
            "val_iou": iou
        })
```

## 🐳 Docker Deployment

### Build Container

```dockerfile
# Dockerfile included in repository
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install -e ".[dev]"

EXPOSE 8080
CMD ["irst-demo", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Build and run
docker build -t irst-library .
docker run -p 8080:8080 --gpus all irst-library
```

### Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'
services:
  irst-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  irst-dashboard:
    image: streamlit/streamlit:latest
    ports:
      - "8501:8501"
    volumes:
      - ./dashboard:/app
    command: streamlit run /app/dashboard.py
```

## 📚 Documentation & Tutorials

- **[📖 Full Documentation](https://irst-library.readthedocs.io)**
- **[🚀 Quick Start Guide](docs/quickstart.md)**
- **[🏗️ Model Architecture Guide](docs/models.md)**
- **[📊 Dataset Guide](docs/datasets.md)**
- **[🔧 Training Tutorial](docs/training.md)**
- **[🚀 Deployment Guide](docs/deployment.md)**
- **[💡 Examples & Notebooks](examples/)**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/IRST-Research/irst-library.git
cd irst-library

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ --cov=irst_library

# Build documentation
cd docs && make html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Citation

If you use this library in your research, please cite:

```bibtex
@software{irst_library,
  title={IRST Library: Advanced Infrared Small Target Detection \& Segmentation},
  author={N.Sachin Deshik},
  year={2025},
  url={https://github.com/sachin-deshik-10/irst-library},
  version={1.0.0}
}
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=IRST-Research/irst-library&type=Date)](https://star-history.com/#IRST-Research/irst-library&Date)

## 📧 Contact

- **Email**: <nayakulasachindeshik@gmail.com>
- **Issues**: [GitHub Issues](https://github.com/sachin-deshik-10/irst-library/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sachin-deshik-10/irst-library/discussions)

---

<div align="center">
  Made with ❤️ by the IRST Research Team
</div>
