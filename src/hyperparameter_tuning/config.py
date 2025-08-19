"""
Hyperparameter configuration management.

This module handles loading, saving, and managing hyperparameter configurations
for all models in the project.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class HyperparameterConfig:
    """
    Comprehensive hyperparameter configuration for neural network models.
    
    Includes model architecture, training, optimization, and gradient descent parameters.
    """
    
    # Model Architecture
    depth: int = 2
    base_channels: int = 32
    dropout: float = 0.3
    num_classes: int = 5
    
    # Training Configuration
    batch_size: int = 32
    num_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Optimizer Configuration
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd', 'rmsprop'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Adam/AdamW specific parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    
    # SGD specific parameters
    momentum: float = 0.9
    nesterov: bool = False
    dampening: float = 0.0
    
    # RMSprop specific parameters
    alpha: float = 0.99
    centered: bool = False
    
    # Learning Rate Scheduler
    scheduler: str = 'plateau'  # 'plateau', 'cosine', 'step', 'exponential', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6
    
    # Gradient Clipping
    gradient_clipping: bool = False
    max_grad_norm: float = 1.0
    
    # Loss Function
    loss_function: str = 'kldiv'  # 'kldiv', 'mse', 'crossentropy'
    
    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    
    # Model specific
    model_name: str = 'promoter_cnn'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyperparameterConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get optimizer-specific parameters."""
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer.lower() in ['adam', 'adamw']:
            base_params.update({
                'betas': (self.beta1, self.beta2),
                'eps': self.eps,
                'amsgrad': self.amsgrad
            })
        elif self.optimizer.lower() == 'sgd':
            base_params.update({
                'momentum': self.momentum,
                'nesterov': self.nesterov,
                'dampening': self.dampening
            })
        elif self.optimizer.lower() == 'rmsprop':
            base_params.update({
                'alpha': self.alpha,
                'eps': self.eps,
                'centered': self.centered,
                'momentum': self.momentum
            })
            
        return base_params
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """Get scheduler-specific parameters."""
        if self.scheduler == 'plateau':
            return {
                'patience': self.scheduler_patience,
                'factor': self.scheduler_factor
            }
        elif self.scheduler == 'step':
            return {
                'step_size': self.scheduler_step_size,
                'gamma': self.scheduler_gamma
            }
        elif self.scheduler == 'exponential':
            return {
                'gamma': self.scheduler_gamma
            }
        elif self.scheduler == 'cosine':
            return {
                'T_max': self.scheduler_t_max,
                'eta_min': self.scheduler_eta_min
            }
        return {}


def get_config_path(model_name: str = 'promoter_cnn') -> Path:
    """Get the path to the configuration file for a specific model."""
    base_dir = Path(__file__).parent
    return base_dir / f'{model_name}_best_config.json'


def save_config(config: HyperparameterConfig, model_name: str = None) -> None:
    """
    Save hyperparameter configuration to JSON file.
    
    Args:
        config: HyperparameterConfig instance to save
        model_name: Optional model name override
    """
    if model_name is None:
        model_name = config.model_name
    
    config_path = get_config_path(model_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"Saved configuration for {model_name} to {config_path}")


def load_config(model_name: str = 'promoter_cnn') -> HyperparameterConfig:
    """
    Load hyperparameter configuration from JSON file.
    
    Args:
        model_name: Name of the model to load config for
        
    Returns:
        HyperparameterConfig instance
    """
    config_path = get_config_path(model_name)
    
    if not config_path.exists():
        print(f"No configuration found for {model_name}, using defaults")
        return HyperparameterConfig(model_name=model_name)
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return HyperparameterConfig.from_dict(config_dict)


def load_best_config(model_name: str = 'promoter_cnn') -> HyperparameterConfig:
    """
    Load the best hyperparameter configuration for a model.
    
    This function first tries to load from the hyperparameter tuning directory,
    then falls back to the notebooks/experiments directory for backward compatibility.
    
    Args:
        model_name: Name of the model to load config for
        
    Returns:
        HyperparameterConfig instance with best parameters
    """
    # Try new location first
    config = load_config(model_name)
    
    # If no config found, try legacy location
    if config.model_name == model_name and not get_config_path(model_name).exists():
        legacy_path = Path(__file__).parent.parent.parent / 'notebooks' / 'experiments' / 'best_hyperparameters.json'
        if legacy_path.exists():
            print(f"Loading legacy configuration from {legacy_path}")
            with open(legacy_path, 'r') as f:
                legacy_config = json.load(f)
            
            # Map legacy config to new format
            config_dict = {
                'model_name': model_name,
                'depth': legacy_config.get('depth', 2),
                'base_channels': legacy_config.get('base_channels', 32),
                'dropout': legacy_config.get('dropout', 0.3),
                'optimizer': legacy_config.get('optimizer', 'adam'),
                'learning_rate': legacy_config.get('lr', 1e-3),
                'weight_decay': legacy_config.get('weight_decay', 1e-4),
                'batch_size': legacy_config.get('batch_size', 32),
                'scheduler': legacy_config.get('scheduler', 'plateau'),
                'scheduler_patience': legacy_config.get('scheduler_patience', 5),
                'scheduler_factor': legacy_config.get('scheduler_factor', 0.5),
                'loss_function': legacy_config.get('loss_function', 'kldiv')
            }
            
            config = HyperparameterConfig.from_dict(config_dict)
            # Save to new location
            save_config(config, model_name)
    
    return config


def update_legacy_config() -> None:
    """Update legacy configuration files to new format."""
    legacy_path = Path(__file__).parent.parent.parent / 'notebooks' / 'experiments' / 'best_hyperparameters.json'
    
    if legacy_path.exists():
        config = load_best_config('promoter_cnn')
        save_config(config, 'promoter_cnn')
        print("Updated legacy configuration to new format")
