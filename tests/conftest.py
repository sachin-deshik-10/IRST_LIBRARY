"""
Test configuration and fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Sample infrared image tensor."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_mask():
    """Sample binary mask tensor."""
    mask = torch.zeros(1, 1, 256, 256)
    # Add a small target
    mask[0, 0, 100:110, 120:130] = 1.0
    return mask


@pytest.fixture
def sample_batch():
    """Sample batch of images and masks."""
    images = torch.randn(4, 1, 256, 256)
    masks = torch.zeros(4, 1, 256, 256)
    
    # Add targets to each image
    for i in range(4):
        x, y = np.random.randint(50, 200, 2)
        masks[i, 0, x:x+10, y:y+10] = 1.0
    
    return {"images": images, "masks": masks}


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory structure."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (temp_dir / "images" / split).mkdir(parents=True)
        (temp_dir / "masks" / split).mkdir(parents=True)
        
        # Create sample files
        for i in range(5):  # 5 samples per split
            # Create sample image
            img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            Image.fromarray(img).save(temp_dir / "images" / split / f"img_{i:03d}.png")
            
            # Create sample mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            if i < 3:  # Add targets to first 3 images
                x, y = np.random.randint(50, 200, 2)
                mask[x:x+10, y:y+10] = 255
            Image.fromarray(mask).save(temp_dir / "masks" / split / f"img_{i:03d}.png")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        "model": {
            "type": "MSHNet",
            "num_classes": 1,
            "backbone": "resnet18",
        },
        "dataset": {
            "type": "SIRSTDataset", 
            "batch_size": 4,
            "num_workers": 0,
        },
        "training": {
            "learning_rate": 0.001,
            "num_epochs": 2,
            "weight_decay": 1e-4,
        },
        "evaluation": {
            "threshold": 0.5,
            "metrics": ["IoU", "Dice"],
        }
    }


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image and mask
        image = torch.randn(1, 256, 256)
        mask = torch.zeros(1, 256, 256)
        
        # Add random target
        if np.random.random() > 0.3:  # 70% chance of having a target
            x, y = np.random.randint(50, 200, 2)
            mask[0, x:x+10, y:y+10] = 1.0
        
        return {
            "image": image,
            "mask": mask,
            "meta": {
                "index": idx,
                "image_name": f"test_{idx:03d}.png",
            }
        }


@pytest.fixture
def mock_dataset():
    """Mock dataset instance."""
    return MockDataset()


# Test markers
pytest.mark.unit = pytest.mark.mark("unit")
pytest.mark.integration = pytest.mark.mark("integration")
pytest.mark.slow = pytest.mark.mark("slow")
