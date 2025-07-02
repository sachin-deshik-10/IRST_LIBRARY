"""
Registry system for models, datasets, and other components.
"""

from typing import Dict, Type, Any, Optional
import inspect


class Registry:
    """A registry to map strings to classes.
    
    This allows for dynamic loading of models, datasets, etc.
    """
    
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}
    
    def __len__(self) -> int:
        return len(self._module_dict)
    
    def __contains__(self, key: str) -> bool:
        return key in self._module_dict
    
    def __repr__(self) -> str:
        format_str = (
            f"{self.__class__.__name__}(name={self._name}, "
            f"items={list(self._module_dict.keys())})"
        )
        return format_str
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict
    
    def get(self, key: str) -> Optional[Type]:
        """Get registered module by key."""
        return self._module_dict.get(key, None)
    
    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ):
        """Register a module.
        
        Args:
            name: Module name. If not provided, use class name.
            force: Whether to override existing module.
            module: Module class to register.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        # Used as a decorator
        if module is None:
            def _register(cls):
                self._register_module(
                    module=cls,
                    module_name=name,
                    force=force
                )
                return cls
            return _register
        
        # Used as a function call
        self._register_module(
            module=module,
            module_name=name,
            force=force
        )
    
    def _register_module(
        self,
        module: Type,
        module_name: Optional[str] = None,
        force: bool = False
    ):
        """Internal method to register a module."""
        if not inspect.isclass(module):
            raise TypeError(f'module must be a class, but got {type(module)}')
        
        if module_name is None:
            module_name = module.__name__
        
        if isinstance(module_name, str):
            module_name = [module_name]
        
        for name in module_name:
            if not force and name in self._module_dict:
                existing_module = self._module_dict[name]
                raise KeyError(
                    f'{name} is already registered in {self.name} at '
                    f'{existing_module.__module__}.{existing_module.__qualname__}'
                )
            self._module_dict[name] = module
    
    def build(self, cfg: Dict[str, Any]) -> Any:
        """Build module from config.
        
        Args:
            cfg: Config dictionary containing module info.
            
        Returns:
            Built module instance.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        
        if 'type' not in cfg:
            if len(cfg) == 1:
                # Assume the only key is the module name
                module_name = list(cfg.keys())[0]
                module_cfg = cfg[module_name]
            else:
                raise KeyError('cfg must contain a "type" key')
        else:
            module_name = cfg.pop('type')
            module_cfg = cfg
        
        if module_name not in self._module_dict:
            raise KeyError(
                f'{module_name} is not in the {self.name} registry. '
                f'Available: {list(self._module_dict.keys())}'
            )
        
        module_cls = self._module_dict[module_name]
        
        try:
            return module_cls(**module_cfg)
        except Exception as e:
            raise RuntimeError(
                f'Failed to build {module_name}: {str(e)}'
            ) from e


# Global registries
MODELS = Registry('models')
DATASETS = Registry('datasets')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
TRANSFORMS = Registry('transforms')
METRICS = Registry('metrics')


def build_from_cfg(cfg: Dict[str, Any], registry: Registry) -> Any:
    """Build a module from config dict.
    
    Args:
        cfg: Config dict.
        registry: Registry to build from.
        
    Returns:
        Built module.
    """
    return registry.build(cfg)


def register_model(name: Optional[str] = None, force: bool = False):
    """Decorator to register models."""
    return MODELS.register_module(name=name, force=force)


def register_dataset(name: Optional[str] = None, force: bool = False):
    """Decorator to register datasets."""
    return DATASETS.register_module(name=name, force=force)


def register_loss(name: Optional[str] = None, force: bool = False):
    """Decorator to register losses."""
    return LOSSES.register_module(name=name, force=force)


def register_metric(name: Optional[str] = None, force: bool = False):
    """Decorator to register metrics."""
    return METRICS.register_module(name=name, force=force)


# Utility functions for getting registered modules
def get_model(name: str) -> Type:
    """Get registered model class by name."""
    model_cls = MODELS.get(name)
    if model_cls is None:
        raise KeyError(f"Model '{name}' not found. Available: {list(MODELS.module_dict.keys())}")
    return model_cls


def get_dataset(name: str) -> Type:
    """Get registered dataset class by name."""
    dataset_cls = DATASETS.get(name)
    if dataset_cls is None:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(DATASETS.module_dict.keys())}")
    return dataset_cls


def list_models() -> list:
    """List all registered models."""
    return list(MODELS.module_dict.keys())


def list_datasets() -> list:
    """List all registered datasets."""
    return list(DATASETS.module_dict.keys())
