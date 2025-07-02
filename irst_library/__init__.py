# Core package structure
from .core import *
from .models import *
from .datasets import *
from .trainers import *
from .evaluation import *

__version__ = "1.0.0"
__author__ = "IRST Research Team"
__email__ = "contact@irst-lib.org"

# Main detector class for easy usage
from .core.detector import IRSTDetector

__all__ = [
    "IRSTDetector",
    "__version__",
    "__author__",
    "__email__",
]
