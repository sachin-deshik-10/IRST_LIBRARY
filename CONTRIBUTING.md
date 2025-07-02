# Contributing to IRST Library

Thank you for your interest in contributing to the IRST Library! This document provides guidelines and instructions for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for training)

### Development Setup

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/sachin-deshik-10/irst-library.git
   cd irst-library
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

5. **Verify Installation**

   ```bash
   pytest tests/ --cov=irst_library
   ```

## 🛠️ Development Workflow

### Branch Naming Convention

- `feature/your-feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/documentation-update` - Documentation updates
- `refactor/code-improvement` - Code refactoring

### Making Changes

1. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests and Linting**

   ```bash
   # Run tests
   pytest tests/ -v

   # Check code formatting
   black --check irst_library/ tests/
   isort --check-only irst_library/ tests/

   # Run type checking
   mypy irst_library/

   # Run linting
   flake8 irst_library/ tests/
   ```

4. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "feat: add new ISTD model implementation"
   ```

5. **Push and Create Pull Request**

   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Naming Conventions

- **Classes**: PascalCase (e.g., `IRSTDetector`)
- **Functions/Variables**: snake_case (e.g., `detect_targets`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_THRESHOLD`)
- **Private methods**: Leading underscore (e.g., `_compute_loss`)

### Documentation

- Use Google-style docstrings
- Include type hints for all functions
- Add inline comments for complex logic

```python
def detect_targets(
    image: torch.Tensor,
    threshold: float = 0.5,
    nms_threshold: float = 0.4
) -> Dict[str, torch.Tensor]:
    """Detect infrared small targets in the input image.
    
    Args:
        image: Input infrared image tensor of shape (B, C, H, W)
        threshold: Detection confidence threshold
        nms_threshold: Non-maximum suppression threshold
        
    Returns:
        Dictionary containing detection results:
            - boxes: Bounding boxes (N, 4)
            - scores: Confidence scores (N,)
            - masks: Segmentation masks (N, H, W)
            
    Raises:
        ValueError: If image tensor has incorrect shape
    """
    # Implementation here
    pass
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_utils.py
├── integration/       # Integration tests
│   ├── test_training.py
│   └── test_inference.py
└── fixtures/          # Test data and fixtures
    ├── sample_images/
    └── configs/
```

### Writing Tests

```python
import pytest
import torch
from irst_library.models import MSHNet

class TestMSHNet:
    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = MSHNet(num_classes=1)
        x = torch.randn(2, 1, 256, 256)
        
        output = model(x)
        
        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = MSHNet(num_classes=1)
        x = torch.randn(batch_size, 1, 256, 256)
        
        output = model(x)
        
        assert output.shape[0] == batch_size
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=irst_library --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## 📚 Adding New Models

### Model Implementation

1. **Create Model File**

   ```python
   # irst_library/models/single_frame/your_model.py
   from typing import Dict, Any
   import torch
   import torch.nn as nn
   from ..base import BaseModel
   from ...core.registry import MODELS

   @MODELS.register_module()
   class YourModel(BaseModel):
       def __init__(self, backbone: str = "resnet18", **kwargs):
           super().__init__()
           self.backbone = self._build_backbone(backbone)
           self.head = self._build_head(**kwargs)
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           features = self.backbone(x)
           return self.head(features)
       
       def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
           # Implement your loss function
           return nn.functional.binary_cross_entropy_with_logits(pred, target)
   ```

2. **Add Configuration**

   ```yaml
   # configs/models/your_model.yaml
   model:
     _target_: irst_library.models.YourModel
     backbone: resnet18
     num_classes: 1
     
   optimizer:
     _target_: torch.optim.Adam
     lr: 0.001
     weight_decay: 1e-4
     
   scheduler:
     _target_: torch.optim.lr_scheduler.StepLR
     step_size: 30
     gamma: 0.1
   ```

3. **Add Tests**

   ```python
   # tests/unit/test_your_model.py
   def test_your_model_creation():
       model = YourModel(backbone="resnet18")
       assert isinstance(model, YourModel)
   ```

4. **Update Documentation**
   - Add model to `docs/models.md`
   - Include performance benchmarks
   - Add usage examples

### Model Checklist

- [ ] Model inherits from `BaseModel`
- [ ] Registered with `@MODELS.register_module()`
- [ ] Implements `forward()` method
- [ ] Implements `compute_loss()` method
- [ ] Has corresponding configuration file
- [ ] Includes comprehensive tests
- [ ] Documentation updated
- [ ] Performance benchmarks included

## 📊 Adding New Datasets

### Dataset Implementation

```python
# irst_library/datasets/your_dataset.py
from typing import Dict, Any, List
from pathlib import Path
import torch
from PIL import Image
from .base import BaseDataset
from ..core.registry import DATASETS

@DATASETS.register_module()
class YourDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Any = None,
        **kwargs
    ):
        super().__init__(root, split, transform)
        self.data_root = Path(root)
        self.image_paths = self._load_image_paths()
        self.mask_paths = self._load_mask_paths()
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self._load_image(idx)
        mask = self._load_mask(idx)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "meta": self._get_meta(idx)
        }
```

### Dataset Checklist

- [ ] Dataset inherits from `BaseDataset`
- [ ] Registered with `@DATASETS.register_module()`
- [ ] Implements `__len__()` and `__getitem__()`
- [ ] Returns consistent dictionary format
- [ ] Includes data loading utilities
- [ ] Has corresponding tests
- [ ] Documentation updated

## 🚀 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**

   ```bash
   pytest tests/ --cov=irst_library
   ```

2. **Check code quality**

   ```bash
   pre-commit run --all-files
   ```

3. **Update documentation**
   - Add/update docstrings
   - Update relevant markdown files
   - Add examples if applicable

4. **Write descriptive commit messages**

   ```
   feat: add SeRankDet model implementation
   
   - Implement selective rank-aware attention mechanism
   - Add configuration files and model weights
   - Include comprehensive tests and benchmarks
   - Update documentation with usage examples
   
   Closes #123
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes generate no new warnings
```

## 📋 Issue Guidelines

### Bug Reports

Use the bug report template and include:

- **Environment details** (OS, Python version, CUDA version)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and stack traces
- **Minimal code example** that reproduces the issue

### Feature Requests

Use the feature request template and include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach
- **Alternative solutions** considered

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** for major feature additions

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Email**: <nayakulasachindeshik@gmail.com> for sensitive issues

## 📄 License

By contributing to IRST Library, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to IRST Library! 🚀
