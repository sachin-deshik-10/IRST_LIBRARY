# Frequently Asked Questions (FAQ)

## General Questions

### What is IRST Library?

IRST Library is a professional, production-ready Python library for Infrared Small Target Detection (ISTD). It provides state-of-the-art deep learning models, comprehensive datasets, and enterprise-grade tools for research and production deployments.

### Who should use IRST Library?

- **Researchers** working on infrared target detection
- **Engineers** developing surveillance and defense systems
- **Data Scientists** building computer vision applications
- **Students** learning about deep learning for infrared imaging

### What makes IRST Library different from other computer vision libraries?

- **Specialized Focus**: Specifically designed for infrared small target detection
- **Production Ready**: Enterprise-grade code quality, documentation, and testing
- **Research Friendly**: Easy integration with popular ML frameworks and experiment tracking
- **Comprehensive**: Includes models, datasets, evaluation metrics, and deployment tools

## Installation and Setup

### What are the system requirements?

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for datasets

### How do I install IRST Library?

```bash
# Basic installation
pip install irst-library

# Development installation
git clone https://github.com/your-org/irst-library
cd irst-library
pip install -e .[dev]

# Docker installation
docker pull irst/library:latest
```

### I'm getting CUDA/GPU errors. What should I do?

1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Set device explicitly: `detector = IRSTDetector(..., device='cpu')`
4. See [GPU Setup Guide](gpu_setup.md) for detailed instructions

### How do I set up the development environment?

```bash
# Use our setup script
python scripts/setup_dev.py

# Or manually:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install
```

## Usage Questions

### How do I get started with a simple detection task?

```python
from irst_library import IRSTDetector

# Load a pretrained model
detector = IRSTDetector.from_pretrained("serank_sirst")

# Detect targets in an image
results = detector.detect("infrared_image.png")
print(f"Found {len(results.targets)} targets")
```

### Which model should I use for my task?

- **SERANKNet**: Best overall performance, good for production
- **ACMNet**: Faster inference, good for real-time applications
- **UNet**: Simple baseline, good for experimentation
- **MSHNet**: Best for multi-scale targets

See our [Model Comparison Guide](model_comparison.md) for detailed benchmarks.

### How do I train on my custom dataset?

```python
from irst_library.training import IRSTTrainer
from irst_library.datasets import CustomDataset

# Prepare your dataset
dataset = CustomDataset(root_dir="path/to/your/data")

# Set up training
trainer = IRSTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train
history = trainer.fit(num_epochs=100)
```

### How do I evaluate model performance?

```python
from irst_library.training.metrics import PixelMetrics, ObjectMetrics

# Pixel-level metrics
pixel_metrics = PixelMetrics()
pixel_metrics.update(predictions, targets)
print(f"IoU: {pixel_metrics.iou():.4f}")

# Object-level metrics
object_metrics = ObjectMetrics()
object_metrics.update(predictions, targets)
print(f"Detection Rate: {object_metrics.detection_rate():.4f}")
```

## Technical Questions

### What data formats are supported?

- **Images**: PNG, JPEG, TIFF, NPY
- **Annotations**: JSON, XML, COCO format
- **Models**: PyTorch (.pth), ONNX (.onnx)

### How do I convert my dataset to the required format?

See our [Dataset Preparation Guide](dataset_preparation.md) for detailed instructions on converting various formats.

### Can I use the library for real-time applications?

Yes! The library supports:
- ONNX export for optimized inference
- TensorRT integration for NVIDIA GPUs
- Model quantization for mobile deployment
- Streaming inference APIs

### How do I deploy models in production?

```python
# Export to ONNX
detector.export("model.onnx", format="onnx")

# Use FastAPI for serving
from irst_library.serving import create_app
app = create_app(model_path="model.onnx")
```

See our [Deployment Guide](deployment.md) for comprehensive instructions.

### What about model interpretability?

```python
from irst_library.utils.visualization import (
    plot_attention_maps,
    generate_gradcam,
    create_feature_visualization
)

# Visualize attention
plot_attention_maps(model, image)

# Generate GradCAM
gradcam = generate_gradcam(model, image, target_layer="backbone.layer4")
```

## Performance and Optimization

### How can I speed up training?

1. **Use mixed precision**: Enable AMP in trainer
2. **Optimize data loading**: Increase `num_workers`, use faster storage
3. **Use multiple GPUs**: Enable DDP training
4. **Optimize hyperparameters**: Use learning rate schedulers, gradient clipping

```python
trainer = IRSTTrainer(
    model=model,
    mixed_precision=True,
    num_workers=8,
    distributed=True
)
```

### My model is overfitting. What should I do?

- Increase regularization (dropout, weight decay)
- Use data augmentation
- Reduce model complexity
- Add early stopping
- Use cross-validation

### How do I monitor training progress?

```python
# Weights & Biases integration
trainer = IRSTTrainer(
    model=model,
    wandb_config={
        'project': 'irst-detection',
        'name': 'experiment-1'
    }
)

# TensorBoard integration
trainer = IRSTTrainer(
    model=model,
    tensorboard_dir='./logs'
)
```

## Troubleshooting

### Common Error Messages

#### "Model not found: {model_name}"
- Check available models: `from irst_library.models import list_models; print(list_models())`
- Verify model name spelling
- Update library: `pip install --upgrade irst-library`

#### "CUDA out of memory"
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use CPU: `device='cpu'`

#### "Dataset not found"
- Check dataset path
- Verify dataset format
- Run dataset validation: `python scripts/validate_dataset.py`

### Performance Issues

#### Training is very slow
- Check GPU utilization: `nvidia-smi`
- Optimize data loading (see performance section)
- Profile your code: `python -m cProfile train.py`

#### Poor model performance
- Check data quality and labels
- Verify preprocessing steps
- Try different models or hyperparameters
- Use more training data

### Getting Help

If you can't find an answer here:

1. **Check the documentation**: [docs/](../docs/)
2. **Search existing issues**: [GitHub Issues](https://github.com/your-org/irst-library/issues)
3. **Ask the community**: [Discussions](https://github.com/your-org/irst-library/discussions)
4. **Report bugs**: [Bug Report Template](https://github.com/your-org/irst-library/issues/new?template=bug_report.md)

## Contributing

### How can I contribute to the project?

- **Report bugs** and feature requests
- **Submit pull requests** with improvements
- **Add new models** or datasets
- **Improve documentation**
- **Help with testing** and validation

See our [Contributing Guide](../CONTRIBUTING.md) for detailed instructions.

### I want to add a new model. What should I do?

1. Follow the [Model Development Guide](model_development.md)
2. Implement the model in `irst_library/models/`
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### How do I report a security vulnerability?

Please see our [Security Policy](../SECURITY.md) for responsible disclosure procedures.

---

**Still have questions?** Feel free to [open an issue](https://github.com/your-org/irst-library/issues/new) or start a [discussion](https://github.com/your-org/irst-library/discussions)!
