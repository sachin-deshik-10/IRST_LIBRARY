"""
Unit tests for models.
"""

import pytest
import torch
from irst_library.models.single_frame.mshnet import MSHNet, SimpleUNet


class TestMSHNet:
    """Test MSHNet model."""
    
    def test_model_creation(self):
        """Test model creation with different configurations."""
        model = MSHNet(backbone="resnet18", num_classes=1)
        assert isinstance(model, MSHNet)
        assert model.num_classes == 1
    
    def test_forward_pass(self):
        """Test forward pass with different input sizes."""
        model = MSHNet()
        
        # Test with standard input
        x = torch.randn(2, 1, 256, 256)
        output = model(x)
        
        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = MSHNet()
        x = torch.randn(batch_size, 1, 128, 128)
        
        output = model(x)
        assert output.shape == (batch_size, 1, 128, 128)
    
    @pytest.mark.parametrize("image_size", [(128, 128), (256, 256), (512, 512)])
    def test_different_image_sizes(self, image_size):
        """Test model with different image sizes."""
        model = MSHNet()
        h, w = image_size
        x = torch.randn(1, 1, h, w)
        
        output = model(x)
        assert output.shape == (1, 1, h, w)
    
    def test_loss_computation(self):
        """Test loss computation."""
        model = MSHNet()
        
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Test different loss types
        bce_loss = model.compute_loss(pred, target, loss_type="bce")
        dice_loss = model.compute_loss(pred, target, loss_type="dice")
        combined_loss = model.compute_loss(pred, target, loss_type="dice_bce")
        
        assert isinstance(bce_loss, torch.Tensor)
        assert isinstance(dice_loss, torch.Tensor)
        assert isinstance(combined_loss, torch.Tensor)
        
        assert bce_loss.requires_grad
        assert dice_loss.requires_grad
        assert combined_loss.requires_grad
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = MSHNet()
        x = torch.randn(1, 1, 128, 128, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 128, 128)).float()
        
        output = model(x)
        loss = model.compute_loss(output, target)
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = MSHNet()
        model.eval()
        
        x = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestSimpleUNet:
    """Test SimpleUNet model."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = SimpleUNet(num_classes=1)
        assert isinstance(model, SimpleUNet)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = SimpleUNet()
        x = torch.randn(2, 1, 128, 128)
        
        output = model(x)
        assert output.shape == (2, 1, 128, 128)
        assert not torch.isnan(output).any()
    
    def test_loss_computation(self):
        """Test loss computation."""
        model = SimpleUNet()
        
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        
        loss = model.compute_loss(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_model_comparison(self):
        """Compare SimpleUNet with MSHNet."""
        simple_model = SimpleUNet()
        msh_model = MSHNet()
        
        # Count parameters
        simple_params = sum(p.numel() for p in simple_model.parameters())
        msh_params = sum(p.numel() for p in msh_model.parameters())
        
        # MSHNet should have more parameters due to multi-scale blocks
        assert msh_params > simple_params


class TestModelRegistry:
    """Test model registration and retrieval."""
    
    def test_model_registration(self):
        """Test that models are properly registered."""
        from irst_library.core.registry import MODELS, get_model
        
        # Check that our models are registered
        assert "mshnet" in MODELS
        assert "simple_unet" in MODELS
        
        # Test getting models
        MSHNetClass = get_model("mshnet")
        SimpleUNetClass = get_model("simple_unet")
        
        assert MSHNetClass == MSHNet
        assert SimpleUNetClass == SimpleUNet
    
    def test_model_instantiation_from_registry(self):
        """Test creating models from registry."""
        from irst_library.core.registry import get_model
        
        MSHNetClass = get_model("mshnet")
        model = MSHNetClass(num_classes=1)
        
        assert isinstance(model, MSHNet)
        
        # Test forward pass
        x = torch.randn(1, 1, 128, 128)
        output = model(x)
        assert output.shape == (1, 1, 128, 128)


@pytest.mark.slow
class TestModelTraining:
    """Test model training scenarios."""
    
    def test_training_step(self):
        """Test a single training step."""
        model = MSHNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate batch
        images = torch.randn(2, 1, 128, 128)
        masks = torch.randint(0, 2, (2, 1, 128, 128)).float()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = model.compute_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_overfitting_single_batch(self):
        """Test that model can overfit a single batch."""
        model = MSHNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Single batch
        images = torch.randn(1, 1, 64, 64)
        masks = torch.zeros(1, 1, 64, 64)
        masks[0, 0, 20:40, 20:40] = 1.0  # Square target
        
        model.train()
        initial_loss = None
        
        # Train for several steps
        for step in range(50):
            optimizer.zero_grad()
            outputs = model(images)
            loss = model.compute_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5
