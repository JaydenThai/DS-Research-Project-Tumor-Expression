#!/usr/bin/env python3
"""
Gradient Descent Focused Hyperparameter Optimization.

This script performs comprehensive hyperparameter optimization with a focus
on gradient descent parameters including optimizer comparison, learning rates,
momentum, Adam parameters, and learning rate schedules.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

# Import required modules
from models.cnn.model import PromoterCNN
from utils.device import DeviceManager
from utils.data import load_data_for_training
from hyperparameter_tuning.tuner import HyperparameterTuner, OptimizationStrategy
from hyperparameter_tuning.search_spaces import SearchSpaceType
from hyperparameter_tuning.config import save_config, HyperparameterConfig


def run_optimizer_comparison(train_dataset, val_dataset, device_manager, num_trials: int = 20):
    """
    Compare different optimizers with various gradient descent parameters.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device_manager: Device management utility
        num_trials: Number of trials per optimizer
    """
    print("🔍 OPTIMIZER COMPARISON STUDY")
    print("=" * 50)
    
    tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager, PromoterCNN)
    
    # Define optimizer-specific search spaces
    optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
    optimizer_results = {}
    
    for optimizer in optimizers:
        print(f"\n📊 Testing {optimizer.upper()} optimizer...")
        
        # Customize search space for this optimizer
        if optimizer in ['adam', 'adamw']:
            search_space = {
                'depth': [2, 3],
                'base_channels': [32],
                'dropout': [0.3],
                'optimizer': [optimizer],
                'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
                'beta1': [0.9, 0.95, 0.99],
                'beta2': [0.999, 0.9999],
                'eps': [1e-8, 1e-7],
                'amsgrad': [False, True],
                'scheduler': ['plateau', 'cosine', 'none'],
                'batch_size': [32, 64],
                'loss_function': ['kldiv']
            }
        elif optimizer == 'sgd':
            search_space = {
                'depth': [2, 3],
                'base_channels': [32],
                'dropout': [0.3],
                'optimizer': [optimizer],
                'learning_rate': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
                'momentum': [0.0, 0.9, 0.95, 0.99],
                'nesterov': [False, True],
                'scheduler': ['plateau', 'step', 'none'],
                'batch_size': [32, 64],
                'loss_function': ['kldiv']
            }
        else:  # rmsprop
            search_space = {
                'depth': [2, 3],
                'base_channels': [32],
                'dropout': [0.3],
                'optimizer': [optimizer],
                'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
                'alpha': [0.9, 0.95, 0.99],
                'centered': [False, True],
                'momentum': [0.0, 0.9],
                'scheduler': ['plateau', 'cosine', 'none'],
                'batch_size': [32, 64],
                'loss_function': ['kldiv']
            }
        
        # Run optimization for this optimizer
        results = []
        for trial in range(num_trials):
            from hyperparameter_tuning.search_spaces import sample_random_config
            config_dict = sample_random_config(search_space)
            result = tuner.evaluate_config(config_dict, max_epochs=25, use_cv=False)
            results.append(result)
            print(f"  Trial {trial+1}/{num_trials}: val_loss={result.val_loss:.4f}")
        
        optimizer_results[optimizer] = results
        
        # Print best result for this optimizer
        best_result = min(results, key=lambda r: r.val_loss)
        print(f"✅ Best {optimizer.upper()}: val_loss={best_result.val_loss:.4f}")
        for key, value in best_result.config.items():
            if key in ['learning_rate', 'weight_decay', 'momentum', 'beta1', 'beta2', 'alpha']:
                print(f"   {key}: {value}")
    
    return optimizer_results


def run_learning_rate_schedule_study(train_dataset, val_dataset, device_manager, num_trials: int = 15):
    """
    Study the effect of different learning rate schedules.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device_manager: Device management utility
        num_trials: Number of trials per scheduler
    """
    print("\n🔍 LEARNING RATE SCHEDULE STUDY")
    print("=" * 50)
    
    tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager, PromoterCNN)
    
    schedulers = ['plateau', 'cosine', 'step', 'exponential', 'none']
    scheduler_results = {}
    
    for scheduler in schedulers:
        print(f"\n📊 Testing {scheduler} scheduler...")
        
        search_space = {
            'depth': [2],
            'base_channels': [32],
            'dropout': [0.3],
            'optimizer': ['adam'],
            'learning_rate': [1e-3, 3e-3],
            'weight_decay': [1e-5, 1e-4],
            'scheduler': [scheduler],
            'batch_size': [64],
            'loss_function': ['kldiv']
        }
        
        # Add scheduler-specific parameters
        if scheduler == 'plateau':
            search_space['scheduler_patience'] = [3, 5, 8]
            search_space['scheduler_factor'] = [0.3, 0.5, 0.7]
        elif scheduler == 'step':
            search_space['scheduler_step_size'] = [5, 10, 15]
            search_space['scheduler_gamma'] = [0.1, 0.3, 0.5]
        elif scheduler == 'exponential':
            search_space['scheduler_gamma'] = [0.9, 0.95, 0.99]
        
        results = []
        for trial in range(num_trials):
            from hyperparameter_tuning.search_spaces import sample_random_config
            config_dict = sample_random_config(search_space)
            result = tuner.evaluate_config(config_dict, max_epochs=30, use_cv=False)
            results.append(result)
            print(f"  Trial {trial+1}/{num_trials}: val_loss={result.val_loss:.4f}")
        
        scheduler_results[scheduler] = results
        
        best_result = min(results, key=lambda r: r.val_loss)
        print(f"✅ Best {scheduler}: val_loss={best_result.val_loss:.4f}")
    
    return scheduler_results


def run_gradient_clipping_study(train_dataset, val_dataset, device_manager, num_trials: int = 10):
    """
    Study the effect of gradient clipping.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device_manager: Device management utility
        num_trials: Number of trials
    """
    print("\n🔍 GRADIENT CLIPPING STUDY")
    print("=" * 50)
    
    tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager, PromoterCNN)
    
    # Test with and without gradient clipping
    clipping_configs = [
        {'gradient_clipping': False},
        {'gradient_clipping': True, 'max_grad_norm': 0.5},
        {'gradient_clipping': True, 'max_grad_norm': 1.0},
        {'gradient_clipping': True, 'max_grad_norm': 2.0},
        {'gradient_clipping': True, 'max_grad_norm': 5.0}
    ]
    
    clipping_results = {}
    
    for clip_config in clipping_configs:
        clip_name = f"clip_{clip_config.get('max_grad_norm', 'none')}"
        print(f"\n📊 Testing gradient clipping: {clip_name}")
        
        search_space = {
            'depth': [2],
            'base_channels': [32],
            'dropout': [0.3],
            'optimizer': ['adam'],
            'learning_rate': [1e-3, 3e-3],
            'weight_decay': [1e-4],
            'scheduler': ['plateau'],
            'batch_size': [64],
            'loss_function': ['kldiv'],
            **clip_config
        }
        
        results = []
        for trial in range(num_trials):
            from hyperparameter_tuning.search_spaces import sample_random_config
            config_dict = sample_random_config(search_space)
            result = tuner.evaluate_config(config_dict, max_epochs=25, use_cv=False)
            results.append(result)
            print(f"  Trial {trial+1}/{num_trials}: val_loss={result.val_loss:.4f}")
        
        clipping_results[clip_name] = results
        
        best_result = min(results, key=lambda r: r.val_loss)
        print(f"✅ Best {clip_name}: val_loss={best_result.val_loss:.4f}")
    
    return clipping_results


def analyze_and_save_results(optimizer_results: Dict, scheduler_results: Dict, 
                           clipping_results: Dict) -> HyperparameterConfig:
    """
    Analyze all results and save the best overall configuration.
    
    Args:
        optimizer_results: Results from optimizer comparison
        scheduler_results: Results from scheduler study
        clipping_results: Results from gradient clipping study
        
    Returns:
        Best hyperparameter configuration
    """
    print("\n📊 ANALYZING RESULTS")
    print("=" * 50)
    
    # Find best result from each study
    all_results = []
    
    for optimizer, results in optimizer_results.items():
        all_results.extend(results)
    
    for scheduler, results in scheduler_results.items():
        all_results.extend(results)
    
    for clip_config, results in clipping_results.items():
        all_results.extend(results)
    
    # Find overall best
    best_result = min(all_results, key=lambda r: r.val_loss)
    
    print(f"🏆 BEST OVERALL CONFIGURATION")
    print(f"   Validation loss: {best_result.val_loss:.4f}")
    print(f"   Training time: {best_result.duration_s:.1f}s")
    print(f"   Epochs trained: {best_result.epochs_trained}")
    
    print(f"\n📋 Best hyperparameters:")
    for key, value in best_result.config.items():
        print(f"   {key}: {value}")
    
    # Create and save best config
    best_config = HyperparameterConfig.from_dict(best_result.config)
    best_config.model_name = 'promoter_cnn'
    save_config(best_config, 'promoter_cnn')
    
    print(f"\n💾 Saved best configuration to promoter_cnn_best_config.json")
    
    return best_config


def plot_optimization_results(optimizer_results: Dict, scheduler_results: Dict, 
                            clipping_results: Dict, save_path: str = None):
    """
    Create visualizations of the optimization results.
    
    Args:
        optimizer_results: Results from optimizer comparison
        scheduler_results: Results from scheduler study
        clipping_results: Results from gradient clipping study
        save_path: Optional path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Optimizer comparison
    ax1 = axes[0, 0]
    optimizer_losses = {}
    for optimizer, results in optimizer_results.items():
        losses = [r.val_loss for r in results]
        optimizer_losses[optimizer] = losses
    
    box_data = [optimizer_losses[opt] for opt in optimizer_losses.keys()]
    box_labels = list(optimizer_losses.keys())
    ax1.boxplot(box_data, labels=box_labels)
    ax1.set_title('Optimizer Comparison')
    ax1.set_ylabel('Validation Loss')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scheduler comparison
    ax2 = axes[0, 1]
    scheduler_losses = {}
    for scheduler, results in scheduler_results.items():
        losses = [r.val_loss for r in results]
        scheduler_losses[scheduler] = losses
    
    box_data = [scheduler_losses[sched] for sched in scheduler_losses.keys()]
    box_labels = list(scheduler_losses.keys())
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_title('Scheduler Comparison')
    ax2.set_ylabel('Validation Loss')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Gradient clipping comparison
    ax3 = axes[1, 0]
    clipping_losses = {}
    for clip_config, results in clipping_results.items():
        losses = [r.val_loss for r in results]
        clipping_losses[clip_config] = losses
    
    box_data = [clipping_losses[clip] for clip in clipping_losses.keys()]
    box_labels = list(clipping_losses.keys())
    ax3.boxplot(box_data, labels=box_labels)
    ax3.set_title('Gradient Clipping Comparison')
    ax3.set_ylabel('Validation Loss')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Learning rate vs validation loss (from all results)
    ax4 = axes[1, 1]
    all_results = []
    for results_dict in [optimizer_results, scheduler_results, clipping_results]:
        for results in results_dict.values():
            all_results.extend(results)
    
    learning_rates = [r.config['learning_rate'] for r in all_results]
    val_losses = [r.val_loss for r in all_results]
    
    ax4.scatter(learning_rates, val_losses, alpha=0.6)
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Validation Loss')
    ax4.set_xscale('log')
    ax4.set_title('Learning Rate vs Validation Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved optimization plots to {save_path}")
    
    plt.show()


def main():
    """Main optimization script."""
    print("🚀 GRADIENT DESCENT HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Initialize device
    device_manager = DeviceManager()
    print(f"Using device: {device_manager.device}")
    
    # Load data
    print("\n📂 Loading data...")
    train_dataset, val_dataset, _ = load_data_for_training()
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Run optimization studies
    print("\n🔬 Running optimization studies...")
    
    # 1. Optimizer comparison
    optimizer_results = run_optimizer_comparison(train_dataset, val_dataset, device_manager, num_trials=15)
    
    # 2. Learning rate schedule study
    scheduler_results = run_learning_rate_schedule_study(train_dataset, val_dataset, device_manager, num_trials=10)
    
    # 3. Gradient clipping study
    clipping_results = run_gradient_clipping_study(train_dataset, val_dataset, device_manager, num_trials=8)
    
    # 4. Analyze and save best configuration
    best_config = analyze_and_save_results(optimizer_results, scheduler_results, clipping_results)
    
    # 5. Create visualizations
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / 'gradient_descent_optimization_results.png'
    plot_optimization_results(optimizer_results, scheduler_results, clipping_results, str(plot_path))
    
    print("\n✅ Gradient descent optimization completed!")
    print(f"   Best validation loss: {min([min([r.val_loss for r in results]) for results in [*optimizer_results.values(), *scheduler_results.values(), *clipping_results.values()]]):.4f}")
    print(f"   Configuration saved to: promoter_cnn_best_config.json")
    

if __name__ == "__main__":
    main()
