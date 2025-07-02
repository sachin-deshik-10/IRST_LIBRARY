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
from .nas import *
from .ssl import *
from .meta_learning import *
from .explainability import *
from .domain_adaptation import *
from .active_learning import *

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
    
    # Neural Architecture Search
    "NASController",
    "SuperNet",
    "EvolutionaryNAS",
    "DifferentiableNAS",
    "NeuralArchitectureSearch",
    "create_nas_experiment",
    "run_nas_search",
    
    # Self-Supervised Learning
    "SimCLR",
    "BYOL",
    "SwAV",
    "MoCo",
    "SelfSupervisedLearning",
    "create_ssl_experiment",
    
    # Meta-Learning
    "MAML",
    "ProtoNet",
    "RelationNet",
    "MetaOptimizer",
    "MetaLearner",
    "create_meta_learning_setup",
    
    # Explainability
    "GradCAM",
    "LIME",
    "SHAP",
    "IntegratedGradients",
    "ExplainabilityAnalyzer",
    "create_explainability_suite",
    
    # Domain Adaptation
    "DANN",
    "CORAL",
    "AdaBN",
    "DomainAdapter",
    "create_domain_adaptation_setup",
    
    # Active Learning
    "UncertaintySampler",
    "CommitteeSampler",
    "DiversitySampler",
    "HybridSampler",
    "ExpectedModelChangeSampler",
    "ParetoActiveLearner",
    "ActiveLearner",
    "StreamingActiveLearner",
    "SamplingStrategy",
    "ActiveLearningConfig",
    "create_active_learning_experiment",
    "benchmark_active_learning_strategies"
]
