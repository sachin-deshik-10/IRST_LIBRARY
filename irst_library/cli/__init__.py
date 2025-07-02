"""
Command-line interface for IRST Library.
"""

import typer
from typing import Optional
from pathlib import Path
import torch

app = typer.Typer(
    name="irst-library",
    help="🔥 Advanced Infrared Small Target Detection Library",
    rich_markup_mode="rich"
)


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    data_root: Optional[str] = typer.Option(None, "--data-root", help="Override data root path"),
    output_dir: Optional[str] = typer.Option("./outputs", "--output", "-o", help="Output directory"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU device ID"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """🚀 Train an ISTD model."""
    try:
        from irst_library.trainers import IRSTTrainer
        from irst_library.utils.config import load_config
        
        typer.echo("Loading configuration...")
        cfg = load_config(config)
        
        # Override config values if provided
        if data_root:
            cfg["dataset"]["root"] = data_root
        if output_dir:
            cfg["training"]["output_dir"] = output_dir
        if gpu is not None:
            cfg["hardware"]["devices"] = [gpu]
        
        typer.echo(f"Starting training with config: {config}")
        
        # Initialize trainer
        trainer = IRSTTrainer(config=cfg)
        
        # Start training
        if resume:
            trainer.fit(resume_from_checkpoint=resume)
        else:
            trainer.fit()
        
        typer.echo("✅ Training completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name or path"),
    output_dir: Optional[str] = typer.Option("./eval_results", "--output", "-o", help="Output directory"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for evaluation"),
    threshold: float = typer.Option(0.5, "--threshold", help="Detection threshold"),
    save_predictions: bool = typer.Option(False, "--save-preds", help="Save prediction visualizations"),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU device ID"),
):
    """📊 Evaluate a trained model."""
    try:
        from irst_library.evaluation import IRSTEvaluator
        
        typer.echo(f"Loading model from: {model_path}")
        typer.echo(f"Evaluating on dataset: {dataset}")
        
        # Set device
        device = f"cuda:{gpu}" if gpu is not None else "auto"
        
        # Initialize evaluator
        evaluator = IRSTEvaluator(
            model_path=model_path,
            dataset_name=dataset,
            device=device,
            threshold=threshold,
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            batch_size=batch_size,
            save_predictions=save_predictions,
            output_dir=output_dir,
        )
        
        # Print results
        typer.echo("📈 Evaluation Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                typer.echo(f"  {metric}: {value:.4f}")
            else:
                typer.echo(f"  {metric}: {value}")
        
        typer.echo("✅ Evaluation completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def benchmark(
    models: str = typer.Option(..., "--models", help="Comma-separated list of model names or paths"),
    datasets: str = typer.Option(..., "--datasets", help="Comma-separated list of dataset names"),
    output: str = typer.Option("benchmark_results.html", "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", help="Output format (html, json, csv)"),
    profile: bool = typer.Option(False, "--profile", help="Profile model efficiency"),
    device: str = typer.Option("auto", "--device", help="Device to use"),
):
    """🏆 Run comprehensive benchmarks."""
    try:
        from irst_library.evaluation import BenchmarkRunner
        
        model_list = [m.strip() for m in models.split(",")]
        dataset_list = [d.strip() for d in datasets.split(",")]
        
        typer.echo(f"Benchmarking {len(model_list)} models on {len(dataset_list)} datasets")
        
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            models=model_list,
            datasets=dataset_list,
            device=device,
            profile_efficiency=profile,
        )
        
        # Run benchmarks
        results = runner.run()
        
        # Save results
        runner.save_results(results, output, format=format)
        
        typer.echo(f"✅ Benchmark completed! Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"❌ Benchmark failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def demo(
    model: str = typer.Option("mshnet", "--model", "-m", help="Model name or path"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Model checkpoint path"),
    host: str = typer.Option("localhost", "--host", help="Host address"),
    port: int = typer.Option(8080, "--port", help="Port number"),
    device: str = typer.Option("auto", "--device", help="Device to use"),
):
    """🎨 Launch interactive demo."""
    try:
        from irst_library.demo import launch_demo
        
        typer.echo(f"Launching demo with model: {model}")
        typer.echo(f"Server will be available at: http://{host}:{port}")
        
        launch_demo(
            model_name=model,
            checkpoint_path=checkpoint,
            host=host,
            port=port,
            device=device,
        )
        
    except Exception as e:
        typer.echo(f"❌ Demo launch failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def export(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    output_path: str = typer.Option(..., "--output", "-o", help="Output path for exported model"),
    format: str = typer.Option("onnx", "--format", help="Export format (onnx, torchscript, tensorrt)"),
    input_size: str = typer.Option("512,512", "--input-size", help="Input image size (H,W)"),
    opset_version: int = typer.Option(11, "--opset", help="ONNX opset version"),
    optimize: bool = typer.Option(True, "--optimize", help="Optimize exported model"),
):
    """📦 Export model for deployment."""
    try:
        from irst_library.deployment import ModelExporter
        
        # Parse input size
        h, w = map(int, input_size.split(","))
        input_shape = (1, 1, h, w)
        
        typer.echo(f"Exporting model from: {model_path}")
        typer.echo(f"Output format: {format}")
        typer.echo(f"Input size: {h}x{w}")
        
        # Initialize exporter
        exporter = ModelExporter(model_path)
        
        # Export model
        if format == "onnx":
            exporter.to_onnx(
                output_path,
                input_shape=input_shape,
                opset_version=opset_version,
                optimize=optimize,
            )
        elif format == "torchscript":
            exporter.to_torchscript(
                output_path,
                input_shape=input_shape,
            )
        elif format == "tensorrt":
            exporter.to_tensorrt(
                output_path,
                input_shape=input_shape,
                optimize=optimize,
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        typer.echo(f"✅ Model exported successfully to: {output_path}")
        
    except Exception as e:
        typer.echo(f"❌ Export failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    model: Optional[str] = typer.Option(None, "--model", help="Show info for specific model"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Show info for specific dataset"),
    list_all: bool = typer.Option(False, "--list", help="List all available models and datasets"),
):
    """ℹ️ Show information about models and datasets."""
    try:
        from irst_library.core.registry import list_models, list_datasets, get_model, get_dataset
        
        if list_all:
            typer.echo("📋 Available Models:")
            for model_name in list_models():
                typer.echo(f"  - {model_name}")
            
            typer.echo("\n📋 Available Datasets:")
            for dataset_name in list_datasets():
                typer.echo(f"  - {dataset_name}")
        
        if model:
            typer.echo(f"🧠 Model Info: {model}")
            model_cls = get_model(model)
            # Create temporary instance to get info
            temp_model = model_cls()
            info = temp_model.get_model_info()
            for key, value in info.items():
                typer.echo(f"  {key}: {value}")
        
        if dataset:
            typer.echo(f"📊 Dataset Info: {dataset}")
            dataset_cls = get_dataset(dataset)
            typer.echo(f"  Class: {dataset_cls.__name__}")
            typer.echo(f"  Module: {dataset_cls.__module__}")
        
    except Exception as e:
        typer.echo(f"❌ Info retrieval failed: {str(e)}", err=True)
        raise typer.Exit(1)


# Entry points for console scripts
def train_cli():
    """Entry point for irst-train command."""
    train()


def evaluate_cli():
    """Entry point for irst-eval command."""
    evaluate()


def benchmark_cli():
    """Entry point for irst-benchmark command."""
    benchmark()


def demo_cli():
    """Entry point for irst-demo command."""
    demo()


if __name__ == "__main__":
    app()
