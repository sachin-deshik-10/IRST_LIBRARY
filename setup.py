#!/usr/bin/env python3
"""
IRST Library - Advanced Infrared Small Target Detection
Setup script for package installation and distribution.
"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    version_file = this_directory / "irst_library" / "__init__.py"
    version_content = version_file.read_text(encoding='utf-8')
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Define requirements
install_requires = [
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "scikit-image>=0.18.0",
    "tqdm>=4.62.0",
    "pyyaml>=6.0",
    "tensorboard>=2.8.0",
    "wandb>=0.12.0",
    "omegaconf>=2.1.0",
    "typer>=0.6.0",
    "rich>=12.0.0",
    "albumentations>=1.2.0",
    "timm>=0.6.0",
    "einops>=0.4.0",
]

dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-mock>=3.7.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pre-commit>=2.17.0",
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
    "nbsphinx>=0.8.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "ipywidgets>=7.7.0",
]

docs_requires = [
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.17.0",
    "myst-parser>=0.17.0",
    "nbsphinx>=0.8.0",
    "pandoc>=2.0.0",
]

deployment_requires = [
    "docker>=5.0.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.12.0",
    "tensorrt>=8.4.0",
    "triton-model-analyzer>=1.19.0",
    "flask>=2.1.0",
    "fastapi>=0.78.0",
    "uvicorn>=0.18.0",
    "gunicorn>=20.1.0",
]

all_requires = install_requires + dev_requires + docs_requires + deployment_requires

setup(
    name="irst-library",
    version=get_version(),
    author="IRST Library Contributors",
    author_email="maintainers@irst-library.org",
    description="Advanced Infrared Small Target Detection Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sachin-deshik-10/irst-library",
    download_url="https://github.com/sachin-deshik-10/irst-library/archive/v2.0.0.tar.gz",
    project_urls={
        "Bug Tracker": "https://github.com/sachin-deshik-10/irst-library/issues",
        "Documentation": "https://irst-library.readthedocs.io",
        "Source Code": "https://github.com/sachin-deshik-10/irst-library",
        "Changelog": "https://github.com/sachin-deshik-10/irst-library/blob/main/CHANGELOG.md",
        "Funding": "https://irst-library.org/funding",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    package_data={
        "irst_library": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "data/*.json",
            "models/pretrained/*.pth",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "deployment": deployment_requires,
        "all": all_requires,
    },
    entry_points={
        "console_scripts": [
            "irst-train=irst_library.cli.train:main",
            "irst-detect=irst_library.cli.detect:main",
            "irst-evaluate=irst_library.cli.evaluate:main",
            "irst-export=irst_library.cli.export:main",
            "irst-benchmark=irst_library.cli.benchmark:main",
            "irst-validate=irst_library.cli.validate:main",
            "irst-deploy=irst_library.cli.deploy:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "infrared",
        "small target detection",
        "computer vision",
        "deep learning",
        "pytorch",
        "surveillance",
        "defense",
        "remote sensing",
        "thermal imaging",
        "object detection",
        "segmentation",
        "artificial intelligence",
        "machine learning",
    ],
    platforms=["any"],
    license="MIT",
    test_suite="tests",
)
