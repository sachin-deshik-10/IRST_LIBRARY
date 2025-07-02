"""
Edge Computing and Mobile Deployment for IRST Library
Optimized inference for resource-constrained environments
"""

from .quantization import ModelQuantizer
from .pruning import ModelPruner
from .distillation import KnowledgeDistiller
from .tensorrt import TensorRTOptimizer
from .mobile import MobileDeployer
from .jetson import JetsonOptimizer

__all__ = [
    "ModelQuantizer",
    "ModelPruner",
    "KnowledgeDistiller",
    "TensorRTOptimizer", 
    "MobileDeployer",
    "JetsonOptimizer"
]
