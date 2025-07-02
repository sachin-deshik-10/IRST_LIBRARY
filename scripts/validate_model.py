#!/usr/bin/env python3
"""Model validation and testing utilities."""

import torch
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ModelValidator:
    """Comprehensive model validation utility."""
    
    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        self.model = model
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        
    def validate_forward_pass(self) -> ValidationResult:
        """Test basic forward pass functionality."""
        try:
            dummy_input = torch.randn(1, *self.input_shape).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
            
            if output is None:
                return ValidationResult(False, "Model returned None")
            
            if not isinstance(output, torch.Tensor):
                return ValidationResult(False, f"Expected tensor output, got {type(output)}")
            
            return ValidationResult(True, "Forward pass successful")
            
        except Exception as e:
            return ValidationResult(False, f"Forward pass failed: {str(e)}")
    
    def validate_output_shape(self, expected_shape: Optional[Tuple[int, ...]] = None) -> ValidationResult:
        """Validate output tensor shape."""
        try:
            dummy_input = torch.randn(1, *self.input_shape).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
            
            actual_shape = output.shape
            
            if expected_shape and actual_shape[1:] != expected_shape:
                return ValidationResult(
                    False, 
                    f"Output shape mismatch. Expected: {expected_shape}, Got: {actual_shape[1:]}"
                )
            
            return ValidationResult(
                True, 
                f"Output shape validation passed: {actual_shape}",
                {"output_shape": actual_shape}
            )
            
        except Exception as e:
            return ValidationResult(False, f"Output shape validation failed: {str(e)}")
    
    def validate_gradient_flow(self) -> ValidationResult:
        """Check if gradients flow properly through the model."""
        try:
            self.model.train()
            dummy_input = torch.randn(1, *self.input_shape, requires_grad=True).to(self.device)
            dummy_target = torch.randn_like(self.model(dummy_input))
            
            loss = torch.nn.functional.mse_loss(self.model(dummy_input), dummy_target)
            loss.backward()
            
            # Check if any parameters have gradients
            has_gradients = any(p.grad is not None for p in self.model.parameters())
            
            if not has_gradients:
                return ValidationResult(False, "No gradients found in model parameters")
            
            # Check for vanishing gradients
            grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
            avg_grad_norm = np.mean(grad_norms)
            
            if avg_grad_norm < 1e-8:
                return ValidationResult(
                    False, 
                    f"Potential vanishing gradients detected. Average gradient norm: {avg_grad_norm}"
                )
            
            return ValidationResult(
                True, 
                "Gradient flow validation passed",
                {"avg_gradient_norm": avg_grad_norm, "gradient_norms": grad_norms}
            )
            
        except Exception as e:
            return ValidationResult(False, f"Gradient flow validation failed: {str(e)}")
    
    def validate_memory_usage(self, max_memory_mb: int = 1000) -> ValidationResult:
        """Validate model memory usage."""
        try:
            if not torch.cuda.is_available():
                return ValidationResult(True, "CUDA not available, skipping memory validation")
            
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Forward pass
            dummy_input = torch.randn(1, *self.input_shape).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            if memory_used > max_memory_mb:
                return ValidationResult(
                    False,
                    f"Memory usage {memory_used:.2f}MB exceeds limit {max_memory_mb}MB"
                )
            
            return ValidationResult(
                True,
                f"Memory usage validation passed: {memory_used:.2f}MB",
                {"memory_used_mb": memory_used, "peak_memory_mb": peak_memory}
            )
            
        except Exception as e:
            return ValidationResult(False, f"Memory validation failed: {str(e)}")
    
    def validate_batch_consistency(self, batch_sizes: List[int] = [1, 4, 8]) -> ValidationResult:
        """Validate model behavior with different batch sizes."""
        try:
            results = {}
            
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, *self.input_shape).to(self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                if output.shape[0] != batch_size:
                    return ValidationResult(
                        False,
                        f"Batch size inconsistency. Input: {batch_size}, Output: {output.shape[0]}"
                    )
                
                results[f"batch_{batch_size}"] = {
                    "input_shape": dummy_input.shape,
                    "output_shape": output.shape
                }
            
            return ValidationResult(
                True,
                "Batch consistency validation passed",
                results
            )
            
        except Exception as e:
            return ValidationResult(False, f"Batch consistency validation failed: {str(e)}")
    
    def validate_reproducibility(self, num_runs: int = 3) -> ValidationResult:
        """Test model output reproducibility with fixed random seed."""
        try:
            outputs = []
            
            for i in range(num_runs):
                torch.manual_seed(42)
                np.random.seed(42)
                
                dummy_input = torch.randn(1, *self.input_shape).to(self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                    outputs.append(output.cpu().numpy())
            
            # Check if all outputs are identical
            for i in range(1, len(outputs)):
                if not np.allclose(outputs[0], outputs[i], rtol=1e-5):
                    return ValidationResult(
                        False,
                        f"Model outputs are not reproducible. Run 0 vs Run {i} differ"
                    )
            
            return ValidationResult(True, "Reproducibility validation passed")
            
        except Exception as e:
            return ValidationResult(False, f"Reproducibility validation failed: {str(e)}")
    
    def validate_numerical_stability(self) -> ValidationResult:
        """Check for numerical stability issues."""
        try:
            # Test with extreme inputs
            test_cases = [
                ("zeros", torch.zeros(1, *self.input_shape)),
                ("ones", torch.ones(1, *self.input_shape)),
                ("large_values", torch.ones(1, *self.input_shape) * 1000),
                ("small_values", torch.ones(1, *self.input_shape) * 1e-6),
            ]
            
            for case_name, test_input in test_cases:
                test_input = test_input.to(self.device)
                
                with torch.no_grad():
                    output = self.model(test_input)
                
                # Check for NaN or Inf values
                if torch.isnan(output).any():
                    return ValidationResult(False, f"NaN values detected with {case_name} input")
                
                if torch.isinf(output).any():
                    return ValidationResult(False, f"Inf values detected with {case_name} input")
            
            return ValidationResult(True, "Numerical stability validation passed")
            
        except Exception as e:
            return ValidationResult(False, f"Numerical stability validation failed: {str(e)}")


def run_comprehensive_validation(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Optional[Tuple[int, ...]] = None,
    max_memory_mb: int = 1000
) -> Dict[str, ValidationResult]:
    """Run comprehensive model validation suite."""
    
    validator = ModelValidator(model, input_shape)
    
    validations = {
        "forward_pass": validator.validate_forward_pass(),
        "output_shape": validator.validate_output_shape(expected_output_shape),
        "gradient_flow": validator.validate_gradient_flow(),
        "memory_usage": validator.validate_memory_usage(max_memory_mb),
        "batch_consistency": validator.validate_batch_consistency(),
        "reproducibility": validator.validate_reproducibility(),
        "numerical_stability": validator.validate_numerical_stability(),
    }
    
    return validations


def main():
    parser = argparse.ArgumentParser(description='Model Validation Utility')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', help='Path to model configuration file')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 256, 256],
                       help='Input shape (without batch dimension)')
    parser.add_argument('--output-file', help='JSON file to save validation results')
    parser.add_argument('--max-memory', type=int, default=1000,
                       help='Maximum memory usage in MB')
    
    args = parser.parse_args()
    
    # Load model (this would need to be adapted based on your model loading logic)
    print(f"Loading model from {args.model_path}")
    try:
        # Example model loading - adapt this to your needs
        model = torch.load(args.model_path, map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # Run validations
    print("Running comprehensive model validation...")
    results = run_comprehensive_validation(
        model=model,
        input_shape=tuple(args.input_shape),
        max_memory_mb=args.max_memory
    )
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{test_name:<20}: {status} - {result.message}")
        if not result.passed:
            all_passed = False
    
    print("="*60)
    print(f"Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Save results to file if requested
    if args.output_file:
        output_data = {
            test_name: {
                "passed": result.passed,
                "message": result.message,
                "details": result.details
            }
            for test_name, result in results.items()
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {args.output_file}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
