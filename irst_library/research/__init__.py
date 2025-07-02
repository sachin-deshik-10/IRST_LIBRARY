"""
Advanced Research Tools for IRST Library
Cutting-edge research capabilities and experimental features
"""

# Advanced implementations
from .quantum_neural import *
from .continual_learning import *
from .physics_informed import *
from .adversarial_robustness import *
from .synthetic_data import *

# Placeholder stubs (to be implemented)
from .nas import NeuralArchitectureSearch
from .ssl import SelfSupervisedLearning
from .meta_learning import MetaLearner
from .explainability import ExplainabilityAnalyzer
from .domain_adaptation import DomainAdapter
from .active_learning import ActiveLearner

__all__ = [
    # Quantum Neural Networks
    "HybridQuantumClassicalNet",
    "QuantumConvLayer",
    "VariationalQuantumClassifier",
    "QuantumInspiredLoss",
    "QuantumTrainer",
    "create_quantum_irst_model",
    
    # Continual Learning
    "ElasticWeightConsolidation",
    "ProgressiveNeuralNetwork", 
    "IntelligentReplayBuffer",
    "ContinualLearningTrainer",
    "create_continual_learning_setup",
    
    # Physics-Informed Neural Networks
    "PhysicsInformedIRSTNet",
    "AtmosphericPropagationLaw",
    "HeatTransferLaw",
    "InfraredRadiationLaw",
    "PhysicsInformedLoss",
    "create_physics_informed_model",
    
    # Adversarial Robustness
    "FastGradientSignMethod",
    "ProjectedGradientDescent",
    "CarliniWagnerAttack",
    "AutoAttack",
    "RandomizedSmoothing",
    "AdversarialTraining",
    "RobustnessEvaluator",
    "create_attack_suite",
    "create_robust_trainer",
    
    # Synthetic Data Generation
    "IRTargetGAN",
    "PhysicsBasedRenderer",
    "DomainRandomizationEngine",
    "ProceduralSyntheticDataset",
    "SyntheticDataPipeline",
    "create_synthetic_dataset",
    "create_synthetic_pipeline",
    
    # Placeholder stubs
    "NeuralArchitectureSearch",
    "SelfSupervisedLearning",
    "MetaLearner", 
    "ExplainabilityAnalyzer",
    "DomainAdapter",
    "ActiveLearner"
]
