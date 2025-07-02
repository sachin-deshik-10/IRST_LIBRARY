#!/usr/bin/env python3
"""Development environment setup script."""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command with error handling."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def setup_virtual_environment():
    """Set up Python virtual environment."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv .venv"):
        return False
    
    print("✓ Virtual environment created")
    return True


def activate_virtualenv():
    """Get activation command for the current platform."""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    
    # Determine pip command
    if platform.system() == "Windows":
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        pip_cmd = ".venv/bin/pip"
    
    commands = [
        f"{pip_cmd} install --upgrade pip",
        f"{pip_cmd} install -e .[dev]",
        f"{pip_cmd} install pre-commit",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            return False
    
    print("✓ Dependencies installed")
    return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("Setting up pre-commit hooks...")
    
    if platform.system() == "Windows":
        precommit_cmd = ".venv\\Scripts\\pre-commit"
    else:
        precommit_cmd = ".venv/bin/pre-commit"
    
    if not run_command(f"{precommit_cmd} install"):
        return False
    
    print("✓ Pre-commit hooks installed")
    return True


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data",
        "experiments",
        "outputs",
        "logs",
        "checkpoints",
        "benchmarks",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        # Create .gitkeep to ensure directory is tracked
        gitkeep = Path(directory) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("✓ Project directories created")


def setup_environment_file():
    """Create .env template file."""
    env_file = Path(".env.template")
    if not env_file.exists():
        env_content = """# IRST Library Environment Configuration Template
# Copy this file to .env and update with your settings

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=irst-library
WANDB_ENTITY=your_username_or_team

# GPU Settings
CUDA_VISIBLE_DEVICES=0

# Data paths
DATA_ROOT=./data
CHECKPOINTS_ROOT=./checkpoints
EXPERIMENTS_ROOT=./experiments

# Development settings
DEBUG=false
LOG_LEVEL=INFO

# Model serving (optional)
MODEL_SERVER_HOST=0.0.0.0
MODEL_SERVER_PORT=8000

# Database (optional)
DATABASE_URL=sqlite:///irst_library.db

# API Keys (optional)
HUGGINGFACE_TOKEN=your_hf_token_here
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✓ Environment template created (.env.template)")


def main():
    """Main setup function."""
    print("🚀 Setting up IRST Library development environment...")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Creating project directories", create_directories),
        ("Creating environment template", setup_environment_file),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Development environment setup complete!")
    print("\nNext steps:")
    print(f"1. Activate virtual environment: {activate_virtualenv()}")
    print("2. Copy .env.template to .env and configure")
    print("3. Run tests: make test")
    print("4. Start coding! 🎉")


if __name__ == "__main__":
    main()
