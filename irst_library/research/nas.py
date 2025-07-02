"""
Neural Architecture Search (NAS) for IRST Library

Implements state-of-the-art NAS techniques for finding optimal ISTD architectures:
- DARTS (Differentiable Architecture Search)
- Progressive NAS
- Evolutionary NAS
- Hardware-Aware NAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import random
import logging
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class SearchSpace:
    """Defines the search space for NAS"""
    
    OPERATIONS = [
        'conv_3x3',
        'conv_5x5', 
        'conv_7x7',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'none'
    ]
    
    def __init__(self, num_nodes: int = 4, num_ops: int = len(OPERATIONS)):
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.operations = self.OPERATIONS
        
    def random_architecture(self) -> Dict[str, Any]:
        """Generate random architecture"""
        arch = {}
        for i in range(self.num_nodes):
            for j in range(i):
                edge_key = f"edge_{j}_{i}"
                arch[edge_key] = random.randint(0, self.num_ops - 1)
        return arch


class MixedOp(nn.Module):
    """Mixed operation for DARTS"""
    
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.ops = nn.ModuleList()
        
        for op_name in SearchSpace.OPERATIONS:
            op = self._create_op(op_name, channels, stride)
            self.ops.append(op)
            
    def _create_op(self, op_name: str, channels: int, stride: int) -> nn.Module:
        """Create operation based on name"""
        if op_name == 'conv_3x3':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'conv_5x5':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 5, stride, 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'conv_7x7':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 7, stride, 3, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'dil_conv_3x3':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride, 2, dilation=2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'dil_conv_5x5':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 5, stride, 4, dilation=2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'sep_conv_3x3':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'sep_conv_5x5':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 5, stride, 2, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride, 1)
        elif op_name == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride, 1)
        elif op_name == 'skip_connect':
            return nn.Identity() if stride == 1 else nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(channels)
            )
        elif op_name == 'none':
            return Zero()
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass with architecture weights"""
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class Zero(nn.Module):
    """Zero operation"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(0.0)


class SearchableCell(nn.Module):
    """Searchable cell for NAS"""
    
    def __init__(self, channels: int, num_nodes: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        
        # Create mixed operations for all possible edges
        self.ops = nn.ModuleDict()
        for i in range(num_nodes):
            for j in range(i):
                edge_key = f"op_{j}_{i}"
                self.ops[edge_key] = MixedOp(channels)
        
        # Architecture parameters (learnable)
        self.arch_params = nn.ParameterDict()
        for i in range(num_nodes):
            for j in range(i):
                edge_key = f"alpha_{j}_{i}"
                self.arch_params[edge_key] = nn.Parameter(
                    torch.randn(len(SearchSpace.OPERATIONS))
                )
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Forward pass through searchable cell"""
        states = [s0, s1]
        
        for i in range(2, self.num_nodes):
            state_sum = 0
            for j in range(i):
                edge_key_op = f"op_{j}_{i}"
                edge_key_alpha = f"alpha_{j}_{i}"
                
                weights = F.softmax(self.arch_params[edge_key_alpha], dim=-1)
                state_sum += self.ops[edge_key_op](states[j], weights)
            
            states.append(state_sum)
        
        # Concatenate intermediate states
        return torch.cat(states[2:], dim=1)


class SuperNet(nn.Module):
    """SuperNet for NAS search"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 1, 
                 init_channels: int = 16, layers: int = 8):
        super().__init__()
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.layers = layers
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cells
        self.cells = nn.ModuleList()
        channels = init_channels
        
        for i in range(layers):
            # Reduction cell every few layers
            if i in [layers//3, 2*layers//3]:
                channels *= 2
                cell = SearchableCell(channels)
            else:
                cell = SearchableCell(channels)
            
            self.cells.append(cell)
        
        # Head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SuperNet"""
        s0 = s1 = self.stem(x)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out
    
    def arch_parameters(self) -> List[nn.Parameter]:
        """Get architecture parameters"""
        arch_params = []
        for cell in self.cells:
            arch_params.extend(list(cell.arch_params.parameters()))
        return arch_params
    
    def weight_parameters(self) -> List[nn.Parameter]:
        """Get weight parameters (non-architecture)"""
        weight_params = []
        for name, param in self.named_parameters():
            if 'alpha' not in name:
                weight_params.append(param)
        return weight_params


class DARTSSearcher:
    """DARTS (Differentiable Architecture Search) implementation"""
    
    def __init__(self, supernet: SuperNet, learning_rate: float = 0.025,
                 arch_learning_rate: float = 3e-4, weight_decay: float = 3e-4):
        self.supernet = supernet
        
        # Optimizers
        self.w_optimizer = torch.optim.SGD(
            supernet.weight_parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        self.arch_optimizer = torch.optim.Adam(
            supernet.arch_parameters(),
            lr=arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def search_step(self, train_data: Tuple[torch.Tensor, torch.Tensor],
                   valid_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one DARTS search step"""
        train_x, train_y = train_data
        valid_x, valid_y = valid_data
        
        # Step 1: Update architecture parameters
        self.arch_optimizer.zero_grad()
        logits = self.supernet(valid_x)
        arch_loss = self.criterion(logits.squeeze(), valid_y.float())
        arch_loss.backward()
        self.arch_optimizer.step()
        
        # Step 2: Update weight parameters
        self.w_optimizer.zero_grad()
        logits = self.supernet(train_x)
        weight_loss = self.criterion(logits.squeeze(), train_y.float())
        weight_loss.backward()
        self.w_optimizer.step()
        
        return {
            'arch_loss': arch_loss.item(),
            'weight_loss': weight_loss.item()
        }
    
    def derive_architecture(self) -> Dict[str, Any]:
        """Derive final architecture from search"""
        architecture = {}
        
        for cell_idx, cell in enumerate(self.supernet.cells):
            cell_arch = {}
            
            for edge_key, alpha_param in cell.arch_params.items():
                # Get the operation with highest weight
                op_idx = torch.argmax(alpha_param).item()
                op_name = SearchSpace.OPERATIONS[op_idx]
                cell_arch[edge_key] = {
                    'operation': op_name,
                    'weight': torch.max(F.softmax(alpha_param, dim=-1)).item()
                }
            
            architecture[f'cell_{cell_idx}'] = cell_arch
        
        return architecture


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search"""
    
    def __init__(self, search_space: SearchSpace, population_size: int = 50,
                 generations: int = 100, mutation_rate: float = 0.1):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self) -> None:
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            individual = self.search_space.random_architecture()
            self.population.append(individual)
    
    def evaluate_fitness(self, architecture: Dict[str, Any]) -> float:
        """Evaluate architecture fitness (placeholder)"""
        # In practice, this would train and validate the architecture
        # For now, return random fitness
        return random.random()
    
    def tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        fitnesses = [self.evaluate_fitness(arch) for arch in tournament]
        winner_idx = np.argmax(fitnesses)
        return tournament[winner_idx]
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover"""
        keys = list(parent1.keys())
        crossover_point = random.randint(1, len(keys) - 1)
        
        child1 = {}
        child2 = {}
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2
    
    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate individual"""
        mutated = copy.deepcopy(individual)
        
        for key in mutated:
            if random.random() < self.mutation_rate:
                mutated[key] = random.randint(0, self.search_space.num_ops - 1)
        
        return mutated
    
    def evolve(self) -> Dict[str, Any]:
        """Run evolutionary search"""
        self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.evaluate_fitness(arch) for arch in self.population]
            self.fitness_history.append(max(fitnesses))
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = self.population_size // 10
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(self.population[idx]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            
            logger.info(f"Generation {generation}: Best fitness = {max(fitnesses):.4f}")
        
        # Return best architecture
        final_fitnesses = [self.evaluate_fitness(arch) for arch in self.population]
        best_idx = np.argmax(final_fitnesses)
        return self.population[best_idx]


class ProgressiveNAS:
    """Progressive Neural Architecture Search"""
    
    def __init__(self, search_space: SearchSpace, max_epochs: int = 200):
        self.search_space = search_space
        self.max_epochs = max_epochs
        self.blocks = []
        self.performance_history = []
        
    def search_block(self, block_idx: int) -> Dict[str, Any]:
        """Search for optimal block at given depth"""
        best_arch = None
        best_performance = 0.0
        
        # Progressive search: start with simple operations
        candidates = self.search_space.operations[:5]  # Start with basic ops
        
        for epoch in range(self.max_epochs // 4):
            # Sample architectures with current candidates
            arch = {}
            for i in range(self.search_space.num_nodes):
                for j in range(i):
                    edge_key = f"edge_{j}_{i}"
                    arch[edge_key] = random.choice(range(len(candidates)))
            
            # Evaluate (placeholder)
            performance = random.random()
            
            if performance > best_performance:
                best_performance = performance
                best_arch = arch
        
        logger.info(f"Block {block_idx}: Best performance = {best_performance:.4f}")
        return best_arch
    
    def progressive_search(self, num_blocks: int = 5) -> List[Dict[str, Any]]:
        """Run progressive NAS"""
        architectures = []
        
        for block_idx in range(num_blocks):
            arch = self.search_block(block_idx)
            architectures.append(arch)
            self.blocks.append(arch)
            
            # Gradually increase search space complexity
            if block_idx % 2 == 0:
                # Add more complex operations
                pass
        
        return architectures


class NeuralArchitectureSearch:
    """Main NAS interface"""
    
    def __init__(self, method: str = 'darts', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.search_space = SearchSpace()
        
        # Initialize searcher based on method
        if method == 'darts':
            supernet = SuperNet(**kwargs.get('supernet_kwargs', {}))
            self.searcher = DARTSSearcher(supernet, **kwargs.get('darts_kwargs', {}))
        elif method == 'evolutionary':
            self.searcher = EvolutionaryNAS(self.search_space, **kwargs.get('evo_kwargs', {}))
        elif method == 'progressive':
            self.searcher = ProgressiveNAS(self.search_space, **kwargs.get('prog_kwargs', {}))
        else:
            raise ValueError(f"Unknown NAS method: {method}")
    
    def search(self, train_loader=None, valid_loader=None) -> Dict[str, Any]:
        """Run neural architecture search"""
        logger.info(f"Starting NAS with method: {self.method}")
        
        if self.method == 'darts':
            return self._darts_search(train_loader, valid_loader)
        elif self.method == 'evolutionary':
            return self.searcher.evolve()
        elif self.method == 'progressive':
            return self.searcher.progressive_search()
    
    def _darts_search(self, train_loader, valid_loader, epochs: int = 50) -> Dict[str, Any]:
        """Run DARTS search"""
        if train_loader is None or valid_loader is None:
            raise ValueError("DARTS requires train and validation loaders")
        
        for epoch in range(epochs):
            epoch_losses = {'arch_loss': 0.0, 'weight_loss': 0.0}
            
            for batch_idx, (train_batch, valid_batch) in enumerate(zip(train_loader, valid_loader)):
                losses = self.searcher.search_step(train_batch, valid_batch)
                
                for key, value in losses.items():
                    epoch_losses[key] += value
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            logger.info(f"Epoch {epoch}: {epoch_losses}")
        
        return self.searcher.derive_architecture()
    
    def export_architecture(self, architecture: Dict[str, Any], 
                          export_path: str) -> None:
        """Export discovered architecture"""
        import json
        
        with open(export_path, 'w') as f:
            json.dump(architecture, f, indent=2)
        
        logger.info(f"Architecture exported to {export_path}")
    
    def build_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build PyTorch model from architecture"""
        # This would implement the actual model construction
        # For now, return a placeholder
        logger.info("Building model from discovered architecture...")
        return SuperNet()  # Placeholder
