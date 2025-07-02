"""
Performance and Load Testing for IRST Library
"""

import pytest
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

from irst_library.models import SERANKNet
from irst_library.datasets import SIRSTDataset


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def model(self):
        """Load model for testing"""
        model = SERANKNet(in_channels=1, num_classes=1)
        model.eval()
        return model
    
    @pytest.fixture  
    def sample_data(self):
        """Generate sample data for testing"""
        return torch.randn(16, 1, 256, 256)
    
    def test_inference_speed_cpu(self, model, sample_data):
        """Test CPU inference speed"""
        model.cpu()
        sample_data = sample_data.cpu()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_data[:1])
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(sample_data[:1])
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Assert reasonable performance (adjust threshold as needed)
        assert avg_time < 0.5, f"CPU inference too slow: {avg_time:.4f}s"
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_speed_gpu(self, model, sample_data):
        """Test GPU inference speed"""
        model.cuda()
        sample_data = sample_data.cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_data[:1])
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(sample_data[:1])
        
        torch.cuda.synchronize()
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Assert reasonable GPU performance
        assert avg_time < 0.1, f"GPU inference too slow: {avg_time:.4f}s"
    
    def test_memory_usage(self, model, sample_data):
        """Test memory usage during inference"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference
        with torch.no_grad():
            _ = model(sample_data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Assert reasonable memory usage (adjust threshold as needed)
        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.2f}MB"
    
    def test_batch_scaling(self, model):
        """Test performance scaling with batch size"""
        batch_sizes = [1, 4, 8, 16, 32]
        times = []
        
        for batch_size in batch_sizes:
            data = torch.randn(batch_size, 1, 256, 256)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(data)
            
            # Benchmark
            start_time = time.time()
            iterations = 20
            
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            times.append(avg_time / batch_size)  # Time per sample
        
        # Assert that per-sample time decreases with larger batches
        assert times[0] > times[-1], "Batching should improve per-sample performance"
    
    def test_concurrent_inference(self, model):
        """Test concurrent inference performance"""
        def run_inference():
            data = torch.randn(1, 1, 256, 256)
            with torch.no_grad():
                return model(data)
        
        # Test with multiple threads
        num_threads = 4
        num_inferences = 20
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_inference) for _ in range(num_inferences)]
            results = [f.result() for f in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert all inferences completed
        assert len(results) == num_inferences
        
        # Assert reasonable concurrent performance
        assert total_time < 10.0, f"Concurrent inference too slow: {total_time:.2f}s"


class TestStressTests:
    """Stress testing for edge cases"""
    
    def test_large_batch_inference(self, model):
        """Test inference with very large batches"""
        try:
            large_data = torch.randn(64, 1, 256, 256)
            with torch.no_grad():
                result = model(large_data)
            assert result.shape[0] == 64
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise
    
    def test_extended_inference(self, model):
        """Test extended inference without memory leaks"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run many inferences
        for i in range(1000):
            data = torch.randn(1, 1, 256, 256)
            with torch.no_grad():
                _ = model(data)
            
            # Check memory every 100 iterations
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Assert no significant memory leaks
                assert memory_increase < 500, f"Potential memory leak: {memory_increase:.2f}MB increase"
