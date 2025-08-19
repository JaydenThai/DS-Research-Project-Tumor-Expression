"""
Search spaces for hyperparameter optimization.

This module defines different search spaces for various optimization strategies
including comprehensive gradient descent parameters.
"""

from enum import Enum
from typing import Dict, Any, List, Union
import numpy as np


class SearchSpaceType(Enum):
    """Types of search spaces for different optimization strategies."""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    GRADIENT_DESCENT_FOCUSED = "gradient_descent_focused"
    ARCHITECTURE_FOCUSED = "architecture_focused"
    QUICK = "quick"


def get_search_space(space_type: SearchSpaceType = SearchSpaceType.COMPREHENSIVE) -> Dict[str, List[Any]]:
    """
    Get hyperparameter search space based on optimization strategy.
    
    Args:
        space_type: Type of search space to return
        
    Returns:
        Dictionary mapping parameter names to lists of possible values
    """
    
    if space_type == SearchSpaceType.BASIC:
        return {
            'depth': [1, 2, 3],
            'base_channels': [16, 32, 48],
            'dropout': [0.2, 0.3, 0.4],
            'optimizer': ['adam'],
            'learning_rate': [1e-3, 3e-3],
            'weight_decay': [1e-5, 1e-4],
            'batch_size': [32, 64],
            'scheduler': ['plateau'],
            'loss_function': ['kldiv', 'mse']
        }
    
    elif space_type == SearchSpaceType.QUICK:
        return {
            'depth': [2],
            'base_channels': [24, 32],
            'dropout': [0.2, 0.3],
            'optimizer': ['adam', 'adamw'],
            'learning_rate': [1e-3, 3e-3],
            'weight_decay': [1e-5, 1e-4],
            'batch_size': [32, 64],
            'scheduler': ['plateau'],
            'loss_function': ['kldiv']
        }
    
    elif space_type == SearchSpaceType.ARCHITECTURE_FOCUSED:
        return {
            'depth': [1, 2, 3, 4, 5],
            'base_channels': [8, 16, 24, 32, 48, 64],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'optimizer': ['adam'],
            'learning_rate': [1e-3],
            'weight_decay': [1e-4],
            'batch_size': [32],
            'scheduler': ['plateau'],
            'loss_function': ['kldiv']
        }
    
    elif space_type == SearchSpaceType.GRADIENT_DESCENT_FOCUSED:
        return {
            'depth': [2, 3],
            'base_channels': [32],
            'dropout': [0.3],
            
            # Optimizer comparison
            'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
            
            # Learning rates
            'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
            
            # Adam/AdamW parameters
            'beta1': [0.9, 0.95, 0.99],
            'beta2': [0.999, 0.9999],
            'eps': [1e-8, 1e-7],
            'amsgrad': [False, True],
            
            # SGD parameters
            'momentum': [0.0, 0.9, 0.95, 0.99],
            'nesterov': [False, True],
            'dampening': [0.0, 0.1],
            
            # RMSprop parameters
            'alpha': [0.9, 0.95, 0.99],
            'centered': [False, True],
            
            # Schedulers
            'scheduler': ['plateau', 'cosine', 'step', 'exponential'],
            'scheduler_patience': [3, 5, 8, 10],
            'scheduler_factor': [0.1, 0.3, 0.5, 0.7],
            'scheduler_step_size': [5, 10, 15, 20],
            'scheduler_gamma': [0.1, 0.3, 0.5, 0.9],
            
            # Gradient clipping
            'gradient_clipping': [False, True],
            'max_grad_norm': [0.5, 1.0, 2.0, 5.0],
            
            'batch_size': [32, 64],
            'loss_function': ['kldiv']
        }
    
    elif space_type == SearchSpaceType.COMPREHENSIVE:
        return {
            # Architecture
            'depth': [1, 2, 3, 4],
            'base_channels': [16, 24, 32, 48, 64],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            
            # Optimizers
            'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
            
            # Learning and regularization
            'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
            
            # Adam/AdamW parameters
            'beta1': [0.9, 0.95],
            'beta2': [0.999, 0.9999],
            'eps': [1e-8, 1e-7],
            'amsgrad': [False, True],
            
            # SGD parameters
            'momentum': [0.0, 0.9, 0.95],
            'nesterov': [False, True],
            'dampening': [0.0, 0.1],
            
            # RMSprop parameters
            'alpha': [0.9, 0.99],
            'centered': [False, True],
            
            # Schedulers
            'scheduler': ['plateau', 'cosine', 'step', 'none'],
            'scheduler_patience': [5, 8, 10],
            'scheduler_factor': [0.3, 0.5, 0.7],
            'scheduler_step_size': [10, 15],
            'scheduler_gamma': [0.1, 0.5],
            
            # Gradient clipping
            'gradient_clipping': [False, True],
            'max_grad_norm': [1.0, 2.0],
            
            # Training
            'batch_size': [16, 32, 64, 128],
            'loss_function': ['kldiv', 'mse'],
            
            # Additional regularization
            'l1_regularization': [0.0, 1e-6, 1e-5],
            'l2_regularization': [0.0, 1e-6, 1e-5]
        }
    
    else:
        raise ValueError(f"Unknown search space type: {space_type}")


def get_continuous_search_space(space_type: SearchSpaceType = SearchSpaceType.COMPREHENSIVE) -> Dict[str, Dict[str, Union[float, List[str]]]]:
    """
    Get continuous search space for Bayesian optimization.
    
    Args:
        space_type: Type of search space to return
        
    Returns:
        Dictionary mapping parameter names to their ranges/choices
    """
    
    base_continuous = {
        'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-1},
        'weight_decay': {'type': 'log_uniform', 'low': 1e-7, 'high': 1e-2},
        'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.6},
        'beta1': {'type': 'uniform', 'low': 0.85, 'high': 0.99},
        'beta2': {'type': 'uniform', 'low': 0.99, 'high': 0.9999},
        'momentum': {'type': 'uniform', 'low': 0.0, 'high': 0.99},
        'alpha': {'type': 'uniform', 'low': 0.85, 'high': 0.99},
        'scheduler_factor': {'type': 'uniform', 'low': 0.1, 'high': 0.8},
        'max_grad_norm': {'type': 'uniform', 'low': 0.5, 'high': 5.0},
        'l1_regularization': {'type': 'log_uniform', 'low': 1e-7, 'high': 1e-4},
        'l2_regularization': {'type': 'log_uniform', 'low': 1e-7, 'high': 1e-4}
    }
    
    categorical_params = {
        'depth': [1, 2, 3, 4],
        'base_channels': [16, 24, 32, 48, 64],
        'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
        'scheduler': ['plateau', 'cosine', 'step', 'none'],
        'batch_size': [16, 32, 64, 128],
        'loss_function': ['kldiv', 'mse'],
        'amsgrad': [False, True],
        'nesterov': [False, True],
        'centered': [False, True],
        'gradient_clipping': [False, True]
    }
    
    # Add categorical parameters
    for param, choices in categorical_params.items():
        base_continuous[param] = {'type': 'categorical', 'choices': choices}
    
    if space_type == SearchSpaceType.QUICK:
        # Reduce search space for quick optimization
        quick_space = {
            'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
            'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-3},
            'dropout': {'type': 'uniform', 'low': 0.2, 'high': 0.4},
            'depth': {'type': 'categorical', 'choices': [2, 3]},
            'base_channels': {'type': 'categorical', 'choices': [24, 32, 48]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw']},
            'batch_size': {'type': 'categorical', 'choices': [32, 64]},
            'loss_function': {'type': 'categorical', 'choices': ['kldiv']}
        }
        return quick_space
    
    elif space_type == SearchSpaceType.GRADIENT_DESCENT_FOCUSED:
        # Focus on gradient descent parameters
        gd_space = base_continuous.copy()
        gd_space['depth'] = {'type': 'categorical', 'choices': [2, 3]}
        gd_space['base_channels'] = {'type': 'categorical', 'choices': [32]}
        gd_space['loss_function'] = {'type': 'categorical', 'choices': ['kldiv']}
        return gd_space
    
    return base_continuous


def sample_random_config(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Sample a random configuration from the search space.
    
    Args:
        search_space: Dictionary mapping parameter names to lists of possible values
        
    Returns:
        Random configuration dictionary
    """
    import random
    
    config = {}
    for param, values in search_space.items():
        config[param] = random.choice(values)
    
    return config


def validate_config_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix configuration parameter compatibility.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and corrected configuration
    """
    validated_config = config.copy()
    
    # SGD with nesterov requires momentum > 0
    if (config.get('optimizer') == 'sgd' and 
        config.get('nesterov', False) and 
        config.get('momentum', 0) == 0):
        validated_config['momentum'] = 0.9
    
    # RMSprop doesn't use beta parameters
    if config.get('optimizer') == 'rmsprop':
        validated_config.pop('beta1', None)
        validated_config.pop('beta2', None)
        validated_config.pop('amsgrad', None)
    
    # SGD doesn't use beta parameters or alpha
    if config.get('optimizer') == 'sgd':
        validated_config.pop('beta1', None)
        validated_config.pop('beta2', None)
        validated_config.pop('amsgrad', None)
        validated_config.pop('alpha', None)
        validated_config.pop('centered', None)
    
    # Adam/AdamW don't use SGD/RMSprop specific parameters
    if config.get('optimizer') in ['adam', 'adamw']:
        validated_config.pop('momentum', None)
        validated_config.pop('nesterov', None)
        validated_config.pop('dampening', None)
        validated_config.pop('alpha', None)
        validated_config.pop('centered', None)
    
    return validated_config
