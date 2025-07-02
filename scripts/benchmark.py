#!/usr/bin/env python3
"""
Professional benchmarking script for IRST Library
Comprehensive performance analysis and model comparison
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.profiler
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

warnings.filterwarnings("ignore")

class IRSTBenchmark:
    """Professional benchmarking suite for IRST models"""
    
    def __init__(self, device: str = "auto", precision: str = "fp32"):
        self.device = self._setup_device(device)
        self.precision = precision
        self.console = Console()
        self.results = {}
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def benchmark_model(
        self,
        model_name: str,
        input_sizes: List[Tuple[int, int]] = None,
        batch_sizes: List[int] = None,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict:
        """Benchmark a specific model"""
        
        if input_sizes is None:
            input_sizes = [(256, 256), (512, 512), (1024, 1024)]
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
            
        self.console.print(f"[bold blue]Benchmarking {model_name}[/bold blue]")
        
        # Load model
        try:
            from irst_library.models import get_model
            model = get_model(model_name).to(self.device)
            model.eval()
            
            if self.precision == "fp16":
                model = model.half()
            elif self.precision == "int8":
                # Add quantization logic here
                pass
                
        except Exception as e:
            self.console.print(f"[red]Failed to load model {model_name}: {e}[/red]")
            return {}
        
        results = {
            "model_name": model_name,
            "device": str(self.device),
            "precision": self.precision,
            "configurations": []
        }
        
        with Progress() as progress:
            task = progress.add_task(f"Benchmarking {model_name}", total=len(input_sizes) * len(batch_sizes))
            
            for input_size in input_sizes:
                for batch_size in batch_sizes:
                    config_results = self._benchmark_configuration(
                        model, batch_size, input_size, num_runs, warmup_runs
                    )
                    results["configurations"].append(config_results)
                    progress.advance(task)
        
        return results
    
    def _benchmark_configuration(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_size: Tuple[int, int],
        num_runs: int,
        warmup_runs: int
    ) -> Dict:
        """Benchmark specific configuration"""
        
        # Create input tensor
        if self.precision == "fp16":
            input_tensor = torch.randn(
                batch_size, 1, *input_size, 
                device=self.device, 
                dtype=torch.float16
            )
        else:
            input_tensor = torch.randn(
                batch_size, 1, *input_size, 
                device=self.device
            )
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Synchronize GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark inference
        inference_times = []
        memory_usage = []
        
        for _ in range(num_runs):
            # Memory before inference
            if self.device.type == "cuda":
                memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # ms
            
            if self.device.type == "cuda":
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        memory_usage = np.array(memory_usage) if memory_usage else np.array([0])
        
        return {
            "batch_size": batch_size,
            "input_size": input_size,
            "inference_time_ms": {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times)),
                "min": float(np.min(inference_times)),
                "max": float(np.max(inference_times)),
                "median": float(np.median(inference_times)),
                "p95": float(np.percentile(inference_times, 95)),
                "p99": float(np.percentile(inference_times, 99))
            },
            "throughput_fps": float(1000 / np.mean(inference_times) * batch_size),
            "memory_usage_mb": {
                "mean": float(np.mean(memory_usage)),
                "std": float(np.std(memory_usage)),
                "max": float(np.max(memory_usage))
            },
            "output_shape": list(output.shape) if hasattr(output, 'shape') else None
        }
    
    def profile_model(self, model_name: str, input_size: Tuple[int, int] = (512, 512)):
        """Profile model with PyTorch profiler"""
        
        try:
            from irst_library.models import get_model
            model = get_model(model_name).to(self.device)
            model.eval()
        except Exception as e:
            self.console.print(f"[red]Failed to load model {model_name}: {e}[/red]")
            return
        
        input_tensor = torch.randn(1, 1, *input_size, device=self.device)
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            for _ in range(10):
                with torch.no_grad():
                    model(input_tensor)
                profiler.step()
        
        # Save profiler results
        profile_path = f"profiles/{model_name}_profile.json"
        Path("profiles").mkdir(exist_ok=True)
        profiler.export_chrome_trace(profile_path)
        
        self.console.print(f"[green]Profile saved to {profile_path}[/green]")
        
        # Print top operations
        self.console.print("\n[bold]Top 10 Operations by CPU Time:[/bold]")
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if self.device.type == "cuda":
            self.console.print("\n[bold]Top 10 Operations by GPU Time:[/bold]")
            print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    def compare_models(self, model_names: List[str], save_results: bool = True) -> Dict:
        """Compare multiple models"""
        
        all_results = {}
        
        for model_name in model_names:
            results = self.benchmark_model(model_name)
            if results:
                all_results[model_name] = results
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        if save_results:
            timestamp = int(time.time())
            results_path = f"benchmark_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            self.console.print(f"[green]Results saved to {results_path}[/green]")
        
        return all_results
    
    def _generate_comparison_report(self, results: Dict):
        """Generate a comparison report table"""
        
        table = Table(title="Model Performance Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Input Size", style="magenta")
        table.add_column("Batch Size", style="yellow")
        table.add_column("Avg Latency (ms)", style="green")
        table.add_column("Throughput (FPS)", style="blue")
        table.add_column("Memory (MB)", style="red")
        
        for model_name, model_results in results.items():
            if not model_results.get("configurations"):
                continue
                
            for config in model_results["configurations"]:
                table.add_row(
                    model_name,
                    f"{config['input_size'][0]}x{config['input_size'][1]}",
                    str(config['batch_size']),
                    f"{config['inference_time_ms']['mean']:.2f}",
                    f"{config['throughput_fps']:.1f}",
                    f"{config['memory_usage_mb']['mean']:.1f}"
                )
        
        self.console.print(table)
    
    def stress_test(self, model_name: str, duration_seconds: int = 300):
        """Run stress test on model"""
        
        try:
            from irst_library.models import get_model
            model = get_model(model_name).to(self.device)
            model.eval()
        except Exception as e:
            self.console.print(f"[red]Failed to load model {model_name}: {e}[/red]")
            return
        
        input_tensor = torch.randn(1, 1, 512, 512, device=self.device)
        
        self.console.print(f"[yellow]Running stress test for {duration_seconds} seconds...[/yellow]")
        
        start_time = time.time()
        iteration_count = 0
        times = []
        
        while time.time() - start_time < duration_seconds:
            iter_start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            iter_end = time.perf_counter()
            times.append((iter_end - iter_start) * 1000)
            iteration_count += 1
        
        # Report results
        times = np.array(times)
        total_time = time.time() - start_time
        
        self.console.print(f"[green]Stress test completed![/green]")
        self.console.print(f"Total iterations: {iteration_count}")
        self.console.print(f"Average FPS: {iteration_count / total_time:.2f}")
        self.console.print(f"Average latency: {np.mean(times):.2f} ms")
        self.console.print(f"Latency std: {np.std(times):.2f} ms")
        self.console.print(f"Max latency: {np.max(times):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="IRST Library Professional Benchmark")
    parser.add_argument("--models", nargs="+", default=["serank", "acm", "unet"],
                       help="Models to benchmark")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--stress", type=int, default=0, help="Stress test duration (seconds)")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    benchmark = IRSTBenchmark(device=args.device, precision=args.precision)
    
    if args.stress > 0:
        for model in args.models:
            benchmark.stress_test(model, args.stress)
    elif args.profile:
        for model in args.models:
            benchmark.profile_model(model)
    else:
        benchmark.compare_models(args.models, save_results=args.save)


if __name__ == "__main__":
    main()
