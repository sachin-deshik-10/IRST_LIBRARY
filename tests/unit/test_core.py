"""
Unit tests for core components.
"""

import pytest
import torch
import numpy as np
from irst_library.core.base import BaseModel, BaseDataset
from irst_library.core.registry import Registry, MODELS, get_model, register_model
from irst_library.core.detector import IRSTDetector


class TestRegistry:
    """Test registry functionality."""
    
    def test_registry_creation(self):
        """Test creating a new registry."""
        registry = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0
    
    def test_module_registration(self):
        """Test registering modules."""
        registry = Registry("test")
        
        @registry.register_module()
        class TestClass:
            pass
        
        assert "TestClass" in registry
        assert len(registry) == 1
        assert registry.get("TestClass") == TestClass
    
    def test_duplicate_registration(self):
        """Test duplicate registration handling."""
        registry = Registry("test")
        
        @registry.register_module()
        class TestClass:
            pass
        
        # Should raise error on duplicate registration
        with pytest.raises(KeyError):
            @registry.register_module()
            class TestClass:  # Same name
                pass
    
    def test_force_registration(self):
        """Test force registration of duplicates."""
        registry = Registry("test")
        
        @registry.register_module()
        class TestClass:
            value = 1
        
        # Force registration should work
        @registry.register_module(force=True)
        class TestClass:  # Same name
            value = 2
        
        assert registry.get("TestClass").value == 2


class TestBaseModel:
    """Test base model functionality."""
    
    def test_base_model_abstract(self):
        """Test that BaseModel is abstract."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_model_implementation(self):
        """Test concrete model implementation."""
        
        class TestModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
            
            def forward(self, x):
                return self.conv(x)
            
            def compute_loss(self, pred, target):
                return torch.nn.functional.mse_loss(pred, target)
        
        model = TestModel()
        assert model.name == "TestModel"
        
        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        output = model(x)
        assert output.shape == x.shape
        
        # Test loss computation
        target = torch.randn_like(output)
        loss = model.compute_loss(output, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_model_info(self):
        """Test model information extraction."""
        
        class SimpleModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 16, 3)
                self.fc = torch.nn.Linear(16, 1)
            
            def forward(self, x):
                return self.fc(self.conv(x).mean(dim=[2, 3]))
            
            def compute_loss(self, pred, target):
                return torch.nn.functional.mse_loss(pred, target)
        
        model = SimpleModel()
        info = model.get_model_info()
        
        assert "name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] == info["total_parameters"]


class TestDetector:
    """Test detector functionality."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        
        # Mock model for testing
        class MockModel(BaseModel):
            def forward(self, x):
                return torch.sigmoid(torch.randn_like(x))
            
            def compute_loss(self, pred, target):
                return torch.nn.functional.binary_cross_entropy(pred, target)
        
        model = MockModel()
        detector = IRSTDetector(model=model, device="cpu")
        
        assert detector.device.type == "cpu"
        assert detector.threshold == 0.5
        assert detector.nms_threshold == 0.4
    
    def test_detector_inference(self):
        """Test detector inference."""
        
        class MockModel(BaseModel):
            def forward(self, x):
                # Return a simple blob in center
                output = torch.zeros_like(x)
                b, c, h, w = x.shape
                center_h, center_w = h // 2, w // 2
                output[:, :, center_h-5:center_h+5, center_w-5:center_w+5] = 0.8
                return output
            
            def compute_loss(self, pred, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        
        model = MockModel()
        detector = IRSTDetector(model=model, device="cpu")
        
        # Test with tensor input
        image = torch.randn(1, 1, 64, 64)
        results = detector.detect(image)
        
        assert "boxes" in results
        assert "scores" in results
        assert "centers" in results
        assert "masks" in results
        
        # Should detect the blob we created
        assert len(results["boxes"]) > 0
    
    def test_detector_batch_inference(self):
        """Test batch inference."""
        
        class MockModel(BaseModel):
            def forward(self, x):
                return torch.sigmoid(torch.randn_like(x) * 0.1)  # Low confidence outputs
            
            def compute_loss(self, pred, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        
        model = MockModel()
        detector = IRSTDetector(model=model, device="cpu", threshold=0.01)  # Low threshold
        
        # Test batch processing
        images = [torch.randn(1, 64, 64) for _ in range(3)]
        results = detector.detect_batch(images, batch_size=2)
        
        assert len(results) == 3
        for result in results:
            assert "boxes" in result
            assert "scores" in result


class MockDataset(BaseDataset):
    """Mock dataset for testing."""
    
    def __init__(self, root="./", split="train", transform=None, num_samples=10):
        super().__init__(root, split, transform)
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(1, 32, 32)
        mask = torch.zeros(1, 32, 32)
        # Add small target
        mask[0, 15:17, 15:17] = 1.0
        
        return {
            "image": image,
            "mask": mask,
            "meta": {"index": idx}
        }


class TestBaseDataset:
    """Test base dataset functionality."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = MockDataset(num_samples=5)
        assert len(dataset) == 5
        assert dataset.split == "train"
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        dataset = MockDataset(num_samples=5)
        sample = dataset[0]
        
        assert "image" in sample
        assert "mask" in sample  
        assert "meta" in sample
        
        assert sample["image"].shape == (1, 32, 32)
        assert sample["mask"].shape == (1, 32, 32)
    
    def test_invalid_split(self):
        """Test invalid split handling."""
        with pytest.raises(ValueError):
            MockDataset(split="invalid")
    
    def test_dataset_info(self):
        """Test dataset information."""
        dataset = MockDataset(num_samples=10)
        info = dataset.get_dataset_info()
        
        assert "name" in info
        assert "num_samples" in info
        assert info["num_samples"] == 10
