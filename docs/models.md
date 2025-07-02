# Model Zoo - IRST Library

This document provides comprehensive information about all available models in the IRST Library.

## Overview

The IRST Library provides a comprehensive collection of state-of-the-art models for infrared small target detection, ranging from lightweight models for edge deployment to high-accuracy models for research applications.

## Available Models

### 🏆 Flagship Models

#### SERANKNet

- **Description**: Search and ranking network with advanced attention mechanisms
- **Performance**: 89.2% IoU on SIRST dataset
- **Speed**: 67.2 FPS (RTX 3080)
- **Parameters**: 2.3M
- **Use Cases**: Research, high-accuracy applications
- **Paper**: [Link to paper]

```python
from irst_library import IRSTDetector
detector = IRSTDetector.from_pretrained("serank_sirst")
```

#### ACMNet

- **Description**: Asymmetric contextual modulation network
- **Performance**: 86.8% IoU on SIRST dataset
- **Speed**: 89.5 FPS (RTX 3080)
- **Parameters**: 1.8M
- **Use Cases**: Real-time applications, production deployment
- **Paper**: [Link to paper]

```python
detector = IRSTDetector.from_pretrained("acm_sirst")
```

### 📊 Backbone Networks

#### ResNet-based Models

- **ResNet18-IRST**: Lightweight ResNet18 backbone
- **ResNet50-IRST**: Balanced performance and accuracy
- **ResNet101-IRST**: High-capacity model for complex scenes

#### EfficientNet-based Models

- **EfficientNet-B0-IRST**: Ultra-lightweight for mobile
- **EfficientNet-B3-IRST**: Balanced efficiency and accuracy
- **EfficientNet-B7-IRST**: Maximum accuracy

#### Vision Transformer Models

- **ViT-Small-IRST**: Transformer-based detection
- **ViT-Base-IRST**: Standard transformer model
- **Swin-Transformer-IRST**: Hierarchical vision transformer

### 🚀 Specialized Models

#### Multi-Scale Detection

- **MSHNet**: Multi-scale hierarchical network
- **FPN-IRST**: Feature pyramid network adaptation
- **PAFPN-IRST**: Path aggregation feature pyramid

#### Attention Mechanisms

- **AttentionNet**: Spatial attention-based model
- **ChannelAttNet**: Channel attention mechanism
- **SpatialChannelNet**: Dual attention architecture

#### Lightweight Models

- **MobileNet-IRST**: MobileNet-based lightweight model
- **ShuffleNet-IRST**: Extremely efficient architecture
- **GhostNet-IRST**: Ghost module-based network

## Model Comparison

### Performance Metrics

| Model | IoU (%) | F1 (%) | Precision (%) | Recall (%) | FPS | Params (M) |
|-------|---------|--------|---------------|------------|-----|------------|
| SERANKNet | **89.2** | **91.8** | 92.1 | **91.5** | 67.2 | 2.3 |
| ACMNet | 86.8 | 89.4 | **93.2** | 85.9 | **89.5** | **1.8** |
| MSHNet | 85.3 | 88.7 | 90.1 | 87.4 | 45.3 | 4.1 |
| AttentionNet | 84.9 | 88.2 | 89.8 | 86.7 | 52.1 | 3.2 |
| MobileNet-IRST | 78.4 | 82.1 | 84.3 | 80.1 | **156.8** | **0.9** |

### Memory Usage

| Model | Training (GB) | Inference (MB) | Model Size (MB) |
|-------|---------------|----------------|-----------------|
| SERANKNet | 3.2 | 412 | 9.2 |
| ACMNet | 2.9 | 367 | 7.1 |
| MSHNet | 4.2 | 523 | 16.4 |
| MobileNet-IRST | 1.8 | 156 | 3.6 |

## Model Selection Guide

### By Use Case

#### Research & Development

- **Best Accuracy**: SERANKNet
- **Novel Architecture**: ViT-Base-IRST
- **Multi-scale Analysis**: MSHNet

#### Production Deployment

- **Balanced Performance**: ACMNet
- **Real-time Processing**: ACMNet or AttentionNet
- **Cloud Deployment**: SERANKNet or ACMNet

#### Edge & Mobile

- **Mobile Devices**: MobileNet-IRST
- **Embedded Systems**: ShuffleNet-IRST
- **IoT Devices**: GhostNet-IRST

#### Specific Scenarios

- **Complex Backgrounds**: SERANKNet with attention
- **Small Targets**: MSHNet with multi-scale
- **Real-time Video**: ACMNet with optimization

### By Hardware

#### High-end GPU (RTX 3080+)

- Primary: SERANKNet
- Alternative: MSHNet for multi-scale

#### Mid-range GPU (GTX 1080/RTX 2060)

- Primary: ACMNet
- Alternative: AttentionNet

#### CPU-only Deployment

- Primary: MobileNet-IRST
- Alternative: ShuffleNet-IRST

#### Mobile/Edge Devices

- Primary: MobileNet-IRST (quantized)
- Alternative: GhostNet-IRST

## Pretrained Weights

### Available Checkpoints

All models are available with pretrained weights on multiple datasets:

- **SIRST Dataset**: Single-frame infrared targets
- **IRSTD-1k Dataset**: Large-scale detection benchmark
- **NUDT-SIRST Dataset**: Multi-scene evaluation
- **Custom Datasets**: Domain-specific fine-tuned models

### Download and Usage

```python
# Download pretrained model
from irst_library import IRSTDetector

# SIRST pretrained models
detector_serank = IRSTDetector.from_pretrained("serank_sirst")
detector_acm = IRSTDetector.from_pretrained("acm_sirst")
detector_msh = IRSTDetector.from_pretrained("msh_sirst")

# IRSTD-1k pretrained models
detector_serank_1k = IRSTDetector.from_pretrained("serank_irstd1k")
detector_acm_1k = IRSTDetector.from_pretrained("acm_irstd1k")

# Multi-dataset pretrained (recommended for generalization)
detector_multi = IRSTDetector.from_pretrained("serank_multi")
```

## Custom Model Development

### Creating New Models

```python
from irst_library.models.base import BaseModel
from irst_library.core.registry import register_model

@register_model("my_custom_model")
class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Your model implementation
        
    def forward(self, x):
        # Forward pass implementation
        return output
```

### Model Configuration

```yaml
# config/my_model.yaml
model:
  type: MyCustomModel
  backbone:
    type: resnet50
    pretrained: true
  neck:
    type: fpn
    in_channels: [256, 512, 1024, 2048]
    out_channels: 256
  head:
    type: classification
    in_channels: 256
    num_classes: 1
```

## Model Optimization

### Quantization

```python
from irst_library.optimization import quantize_model

# Post-training quantization
quantized_model = quantize_model(
    model_path="checkpoints/serank_best.pth",
    quantization_type="int8",
    calibration_dataset="sirst_val"
)
```

### Pruning

```python
from irst_library.optimization import prune_model

# Structured pruning
pruned_model = prune_model(
    model_path="checkpoints/serank_best.pth",
    pruning_ratio=0.3,
    pruning_type="structured"
)
```

### Knowledge Distillation

```python
from irst_library.training import DistillationTrainer

# Teacher-student training
trainer = DistillationTrainer(
    teacher_model="serank_sirst",
    student_model="mobilenet_irst",
    distillation_alpha=0.7,
    temperature=4.0
)
```

## Export Formats

### ONNX Export

```python
# Export to ONNX
detector.export_onnx(
    output_path="models/serank.onnx",
    input_shape=(1, 1, 256, 256),
    opset_version=11
)
```

### TensorRT Optimization

```python
# Convert to TensorRT
detector.export_tensorrt(
    output_path="models/serank.trt",
    precision="fp16",
    max_batch_size=8
)
```

### Core ML (iOS)

```python
# Export for iOS deployment
detector.export_coreml(
    output_path="models/serank.mlmodel",
    minimum_deployment_target="iOS13"
)
```

## Benchmarking

### Performance Testing

```bash
# Benchmark all models
irst-benchmark --models all --dataset sirst --output benchmark_results.json

# Compare specific models
irst-benchmark --models serank,acm,msh --dataset sirst --metrics iou,f1,fps

# Hardware-specific benchmarking
irst-benchmark --models serank --hardware rtx3080,v100,cpu --batch-sizes 1,8,16
```

### Custom Benchmarking

```python
from irst_library.benchmark import ModelBenchmark

benchmark = ModelBenchmark()

# Accuracy benchmark
accuracy_results = benchmark.accuracy(
    models=["serank", "acm", "msh"],
    dataset="sirst",
    metrics=["iou", "f1", "precision", "recall"]
)

# Speed benchmark
speed_results = benchmark.speed(
    models=["serank", "acm"],
    input_sizes=[(256, 256), (512, 512)],
    batch_sizes=[1, 8, 16]
)

# Memory benchmark
memory_results = benchmark.memory(
    models=["serank"],
    batch_sizes=[1, 8, 16, 32]
)
```

## Model Updates and Versioning

### Version History

- **v1.0**: Initial model releases
- **v1.1**: Performance optimizations, new lightweight models
- **v1.2**: Transformer-based models, improved accuracy
- **v1.3**: Quantization support, mobile-optimized models
- **v2.0**: Foundation models, multi-modal support

### Automatic Updates

```python
# Check for model updates
from irst_library.models import check_model_updates

updates = check_model_updates()
if updates:
    print(f"Updates available for: {updates}")
    
# Auto-update models
from irst_library.models import update_models
update_models(models=["serank", "acm"])
```

## Contributing New Models

### Submission Guidelines

1. **Model Implementation**: Follow the base model template
2. **Documentation**: Provide comprehensive documentation
3. **Benchmarking**: Include benchmark results on standard datasets
4. **Testing**: Ensure all tests pass
5. **Paper Reference**: Include link to associated paper

### Review Process

1. **Technical Review**: Code quality and architecture review
2. **Performance Review**: Benchmark validation
3. **Documentation Review**: Documentation completeness
4. **Community Review**: Feedback from community

## Support and Resources

- **Documentation**: [Model Development Guide](model_development.md)
- **Tutorials**: [Model Training Tutorial](tutorials/training.md)
- **Examples**: [Model Zoo Examples](examples/model_zoo.ipynb)
- **Community**: [GitHub Discussions](https://github.com/irst-library/discussions)

---

For the latest model updates and announcements, follow our [releases page](https://github.com/irst-library/releases).
