# Dataset Guide - IRST Library

This comprehensive guide covers all aspects of dataset management, preparation, and usage in the IRST Library.

## Supported Datasets

### 🎯 Primary Datasets

#### SIRST Dataset

- **Description**: Single-frame infrared small target dataset
- **Size**: 427 training images, 142 test images
- **Resolution**: Variable (256×256 to 512×512)
- **Format**: PNG images with binary masks
- **Download**: [Official Link](https://github.com/YimianDai/sirst)

```python
from irst_library.datasets import SIRSTDataset

dataset = SIRSTDataset(
    root="./data/SIRST",
    split="train",
    transform=None
)
```

#### IRSTD-1k Dataset

- **Description**: Large-scale infrared small target detection
- **Size**: 1,000 high-quality images
- **Resolution**: 512×512
- **Format**: TIFF images with XML annotations
- **Download**: [Dataset Portal](https://link-to-dataset)

```python
from irst_library.datasets import IRSTD1kDataset

dataset = IRSTD1kDataset(
    root="./data/IRSTD-1k",
    split="train",
    image_size=(512, 512)
)
```

#### NUDT-SIRST Dataset

- **Description**: Multi-scene infrared target detection
- **Size**: 10,000+ images across multiple scenarios
- **Resolution**: Variable
- **Format**: Multiple formats supported
- **Download**: [University Portal](https://link-to-nudt)

```python
from irst_library.datasets import NUDTSIRSTDataset

dataset = NUDTSIRSTDataset(
    root="./data/NUDT-SIRST",
    split="train",
    scenario="all"  # or specific: urban, rural, marine
)
```

### 🌟 Specialized Datasets

#### Multi-Modal Datasets

- **IR-RGB-Paired**: Infrared and RGB paired dataset
- **Thermal-Visible**: Thermal and visible spectrum pairs
- **Multi-Spectral**: Multi-spectral infrared imaging

#### Synthetic Datasets

- **SyntheticIR**: Procedurally generated infrared scenes
- **SimulatedTargets**: Physics-based target simulation
- **AugmentedReal**: Real data with synthetic targets

## Dataset Preparation

### Directory Structure

```
data/
├── SIRST/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── masks/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── IRSTD-1k/
│   ├── images/
│   ├── annotations/
│   └── metadata.json
└── custom_dataset/
    ├── images/
    ├── labels/
    └── config.yaml
```

### Data Format Requirements

#### Image Formats

- **Supported**: PNG, JPEG, TIFF, NPY, HDF5
- **Recommended**: PNG for lossless compression
- **Bit Depth**: 8-bit, 16-bit supported
- **Channels**: Grayscale (1), RGB (3), Multi-spectral (N)

#### Annotation Formats

- **Segmentation Masks**: Binary PNG masks
- **Bounding Boxes**: COCO JSON format
- **Points**: CSV with x,y coordinates
- **Custom**: YAML/JSON with metadata

### Data Validation

```python
from irst_library.datasets.utils import validate_dataset

# Validate dataset structure and format
validation_report = validate_dataset(
    dataset_path="./data/SIRST",
    dataset_type="sirst",
    check_images=True,
    check_annotations=True,
    generate_report=True
)

print(f"Validation Status: {validation_report['status']}")
print(f"Issues Found: {len(validation_report['issues'])}")
```

## Data Preprocessing

### Standard Pipeline

```python
import albumentations as A
from irst_library.datasets.transforms import IRSTTransforms

# Basic preprocessing pipeline
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485], std=[0.229]),
    IRSTTransforms.ToTensor()
])

dataset = SIRSTDataset(
    root="./data/SIRST",
    split="train",
    transform=transform
)
```

### Advanced Augmentation

```python
# Advanced augmentation for training
train_transform = A.Compose([
    # Geometric transformations
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.7
    ),
    
    # Photometric augmentations
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.6
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.3),
    
    # Infrared-specific augmentations
    IRSTTransforms.ThermalNoise(intensity=0.1, p=0.3),
    IRSTTransforms.AtmosphericDistortion(strength=0.2, p=0.4),
    IRSTTransforms.SensorNoise(noise_type="poisson", p=0.3),
    
    # Final preprocessing
    A.Resize(256, 256),
    A.Normalize(mean=[0.485], std=[0.229]),
    IRSTTransforms.ToTensor()
])
```

### Multi-Scale Training

```python
from irst_library.datasets.transforms import MultiScaleTransform

# Multi-scale training strategy
multiscale_transform = MultiScaleTransform(
    scales=[224, 256, 320, 384],
    scale_probability=[0.2, 0.4, 0.3, 0.1],
    interpolation="bilinear"
)

dataset = SIRSTDataset(
    root="./data/SIRST",
    split="train",
    transform=multiscale_transform
)
```

## Custom Dataset Creation

### Dataset Class Template

```python
from irst_library.datasets.base import BaseDataset
from irst_library.core.registry import register_dataset
import cv2
import json
from pathlib import Path

@register_dataset("my_custom_dataset")
class MyCustomDataset(BaseDataset):
    """Custom dataset implementation."""
    
    def __init__(self, root, split="train", transform=None, **kwargs):
        super().__init__(root, split, transform)
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Load dataset information
        self.data_info = self._load_data_info()
        
    def _load_data_info(self):
        """Load dataset information from metadata files."""
        metadata_file = self.root / f"{self.split}.json"
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        """Get dataset item."""
        item_info = self.data_info[idx]
        
        # Load image
        image_path = self.root / "images" / item_info["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask_path = self.root / "masks" / item_info["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Prepare sample
        sample = {
            "image": image,
            "mask": mask,
            "metadata": {
                "image_id": item_info["id"],
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": self.split
            }
        }
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_class_weights(self):
        """Calculate class weights for balanced training."""
        positive_pixels = 0
        total_pixels = 0
        
        for idx in range(len(self)):
            sample = self.__getitem__(idx)
            mask = sample["mask"]
            positive_pixels += (mask > 0).sum()
            total_pixels += mask.numel()
        
        pos_weight = (total_pixels - positive_pixels) / positive_pixels
        return torch.tensor([1.0, pos_weight])
```

### Dataset Configuration

```yaml
# custom_dataset_config.yaml
dataset:
  name: "my_custom_dataset"
  root: "./data/custom"
  splits:
    train: "train.json"
    val: "val.json"
    test: "test.json"
  
  preprocessing:
    resize: [256, 256]
    normalize:
      mean: [0.485]
      std: [0.229]
  
  augmentation:
    rotation: 15
    flip_horizontal: 0.5
    flip_vertical: 0.3
    brightness: 0.2
    contrast: 0.2
    noise: 0.1
  
  class_info:
    background: 0
    target: 1
    
  metadata:
    description: "Custom infrared small target dataset"
    version: "1.0"
    author: "Your Name"
    license: "MIT"
```

## Data Loading and Batching

### Standard Data Loader

```python
from torch.utils.data import DataLoader
from irst_library.datasets.collate import irst_collate_fn

# Standard data loader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=irst_collate_fn,
    pin_memory=True,
    drop_last=True
)
```

### Advanced Data Loading

```python
from irst_library.datasets.samplers import (
    BalancedSampler,
    MultiScaleSampler,
    DistributedSampler
)

# Balanced sampling for imbalanced datasets
balanced_sampler = BalancedSampler(
    dataset,
    samples_per_class=1000,
    replacement=True
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    sampler=balanced_sampler,
    num_workers=4
)

# Multi-scale sampling
multiscale_sampler = MultiScaleSampler(
    dataset,
    scales=[224, 256, 320],
    batch_size=16
)
```

### Distributed Training Setup

```python
from torch.utils.data.distributed import DistributedSampler

# Distributed data loading
if torch.distributed.is_initialized():
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True
    )
else:
    sampler = None

dataloader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)
```

## Dataset Analysis and Statistics

### Dataset Statistics

```python
from irst_library.datasets.analysis import DatasetAnalyzer

analyzer = DatasetAnalyzer(dataset)

# Basic statistics
stats = analyzer.compute_statistics()
print(f"Dataset size: {stats['total_samples']}")
print(f"Average image size: {stats['avg_image_size']}")
print(f"Target density: {stats['target_density']:.4f}")
print(f"Class distribution: {stats['class_distribution']}")

# Advanced analysis
distribution_analysis = analyzer.analyze_distribution()
size_analysis = analyzer.analyze_target_sizes()
contrast_analysis = analyzer.analyze_contrast()
```

### Visualization

```python
from irst_library.datasets.visualization import DatasetVisualizer

visualizer = DatasetVisualizer(dataset)

# Dataset overview
visualizer.plot_overview(save_path="dataset_overview.png")

# Class distribution
visualizer.plot_class_distribution(save_path="class_dist.png")

# Target size distribution
visualizer.plot_target_sizes(save_path="target_sizes.png")

# Sample visualization
visualizer.plot_samples(
    num_samples=16,
    save_path="sample_images.png"
)
```

## Data Quality Assessment

### Quality Metrics

```python
from irst_library.datasets.quality import QualityAssessment

qa = QualityAssessment(dataset)

# Image quality metrics
quality_report = qa.assess_image_quality()
print(f"Average PSNR: {quality_report['avg_psnr']:.2f}")
print(f"Average SSIM: {quality_report['avg_ssim']:.4f}")
print(f"Blur detection: {quality_report['blur_ratio']:.2%}")

# Annotation quality
annotation_report = qa.assess_annotation_quality()
print(f"Annotation consistency: {annotation_report['consistency']:.4f}")
print(f"Missing annotations: {annotation_report['missing_count']}")
```

### Data Cleaning

```python
from irst_library.datasets.cleaning import DataCleaner

cleaner = DataCleaner(dataset)

# Remove duplicates
duplicates = cleaner.find_duplicates(threshold=0.95)
print(f"Found {len(duplicates)} duplicate images")

# Fix annotation issues
fixed_annotations = cleaner.fix_annotations()
print(f"Fixed {len(fixed_annotations)} annotation issues")

# Remove low-quality samples
quality_filtered = cleaner.filter_by_quality(
    min_psnr=20,
    min_ssim=0.7,
    max_blur_ratio=0.1
)
```

## Performance Optimization

### Data Loading Optimization

```python
# Optimized data loading configuration
def create_optimized_dataloader(dataset, batch_size=16):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
```

### Memory-Efficient Loading

```python
from irst_library.datasets.efficient import MemoryEfficientDataset

# Memory-efficient dataset for large datasets
efficient_dataset = MemoryEfficientDataset(
    dataset_path="./data/large_dataset",
    cache_size=1000,  # Cache 1000 images in memory
    preprocess_on_demand=True
)
```

### Preprocessing Caching

```python
from irst_library.datasets.cache import PreprocessingCache

# Cache preprocessed data
cache = PreprocessingCache(
    cache_dir="./cache",
    max_cache_size_gb=10
)

dataset = SIRSTDataset(
    root="./data/SIRST",
    split="train",
    transform=transform,
    cache=cache
)
```

## Dataset Utilities

### Dataset Conversion

```python
from irst_library.datasets.converters import (
    COCOToIRST,
    YOLOToIRST,
    PascalVOCToIRST
)

# Convert COCO format to IRST format
converter = COCOToIRST()
converter.convert(
    input_path="./data/coco_dataset",
    output_path="./data/irst_dataset",
    split_ratios=[0.7, 0.2, 0.1]  # train, val, test
)
```

### Dataset Splitting

```python
from irst_library.datasets.splitting import DatasetSplitter

splitter = DatasetSplitter(dataset)

# Random splitting
train_dataset, val_dataset, test_dataset = splitter.random_split(
    ratios=[0.7, 0.2, 0.1],
    seed=42
)

# Stratified splitting (maintains class distribution)
train_dataset, val_dataset, test_dataset = splitter.stratified_split(
    ratios=[0.7, 0.2, 0.1],
    stratify_by="target_count"
)
```

### Dataset Merging

```python
from irst_library.datasets.merging import DatasetMerger

merger = DatasetMerger()

# Merge multiple datasets
merged_dataset = merger.merge_datasets([
    dataset1,
    dataset2,
    dataset3
], strategy="concatenate")

# Advanced merging with harmonization
harmonized_dataset = merger.merge_with_harmonization([
    dataset1,
    dataset2
], harmonization_strategy="histogram_matching")
```

## Best Practices

### Data Preparation Checklist

- ✅ **Consistent Format**: Ensure all images have consistent format and naming
- ✅ **Quality Check**: Remove low-quality or corrupted images
- ✅ **Balanced Distribution**: Consider class imbalance and target size distribution
- ✅ **Proper Splitting**: Use stratified splitting for representative splits
- ✅ **Annotation Validation**: Verify annotation accuracy and completeness
- ✅ **Metadata Documentation**: Include comprehensive metadata

### Performance Tips

- 🚀 **Use SSD Storage**: Store datasets on fast SSD for better I/O performance
- 🚀 **Optimize Workers**: Set `num_workers` to 2×CPU cores for optimal performance
- 🚀 **Enable Pin Memory**: Use `pin_memory=True` for GPU training
- 🚀 **Cache Preprocessing**: Cache preprocessed data for repeated use
- 🚀 **Batch Size Tuning**: Find optimal batch size for your hardware

### Common Pitfalls to Avoid

- ❌ **Data Leakage**: Ensure no overlap between train/val/test splits
- ❌ **Inconsistent Preprocessing**: Use same preprocessing for train and inference
- ❌ **Memory Issues**: Monitor memory usage with large datasets
- ❌ **Poor Augmentation**: Avoid unrealistic augmentations for infrared data
- ❌ **Missing Normalization**: Always normalize input data appropriately

## Support and Resources

- **Dataset Issues**: [GitHub Issues](https://github.com/irst-library/issues)
- **Data Contributions**: [Contribution Guide](../CONTRIBUTING.md)
- **Format Questions**: [Community Forum](https://github.com/irst-library/discussions)
- **Custom Datasets**: [Dataset Development Tutorial](tutorials/custom_datasets.md)
