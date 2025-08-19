"""
Comprehensive hyperparameter tuning implementation.

This module provides advanced hyperparameter optimization strategies including
random search, grid search, and Bayesian optimization with extensive
gradient descent parameter tuning.
"""

import time
import random
import itertools
import pickle
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold

from config import HyperparameterConfig, save_config
from search_spaces import get_search_space, get_continuous_search_space, SearchSpaceType, sample_random_config, validate_config_compatibility


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


@dataclass
class TrialResult:
    """Data class to store hyperparameter trial results."""
    config: Dict[str, Any]
    val_loss: float
    val_losses: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    params: int = 0
    duration_s: float = 0.0
    epochs_trained: int = 0
    final_lr: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    gradient_norms: List[float] = field(default_factory=list)
    optimizer_state: Dict = field(default_factory=dict)


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for neural network models.
    
    Supports multiple optimization strategies with extensive gradient descent
    parameter exploration.
    """
    
    def __init__(self, train_dataset, val_dataset, device_manager, model_class):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            device_manager: Device management utility
            model_class: Model class to instantiate
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device_manager.device
        self.model_class = model_class
        self.results = []
        
    def create_optimizer(self, model: nn.Module, config: HyperparameterConfig) -> optim.Optimizer:
        """
        Create optimizer based on configuration with full parameter support.
        
        Args:
            model: PyTorch model
            config: Hyperparameter configuration
            
        Returns:
            Configured optimizer
        """
        optimizer_params = config.get_optimizer_params()
        
        if config.optimizer.lower() == 'adam':
            return optim.Adam(model.parameters(), **optimizer_params)
        elif config.optimizer.lower() == 'adamw':
            return optim.AdamW(model.parameters(), **optimizer_params)
        elif config.optimizer.lower() == 'sgd':
            return optim.SGD(model.parameters(), **optimizer_params)
        elif config.optimizer.lower() == 'rmsprop':
            return optim.RMSprop(model.parameters(), **optimizer_params)
        else:
            print(f"Unknown optimizer {config.optimizer}, defaulting to Adam")
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    def create_scheduler(self, optimizer: optim.Optimizer, config: HyperparameterConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: PyTorch optimizer
            config: Hyperparameter configuration
            
        Returns:
            Configured scheduler or None
        """
        if config.scheduler == 'none':
            return None
            
        scheduler_params = config.get_scheduler_params()
        
        if config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        elif config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif config.scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        else:
            print(f"Unknown scheduler {config.scheduler}, using plateau")
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    def create_criterion(self, config: HyperparameterConfig) -> nn.Module:
        """
        Create loss function based on configuration.
        
        Args:
            config: Hyperparameter configuration
            
        Returns:
            Loss function
        """
        if config.loss_function == 'kldiv':
            return nn.KLDivLoss(reduction='batchmean')
        elif config.loss_function == 'mse':
            return nn.MSELoss()
        elif config.loss_function == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            print(f"Unknown loss function {config.loss_function}, using KLDivLoss")
            return nn.KLDivLoss(reduction='batchmean')
    
    def compute_regularization_loss(self, model: nn.Module, config: HyperparameterConfig) -> torch.Tensor:
        """
        Compute additional regularization losses.
        
        Args:
            model: PyTorch model
            config: Hyperparameter configuration
            
        Returns:
            Regularization loss tensor
        """
        reg_loss = 0.0
        
        if config.l1_regularization > 0:
            l1_loss = 0.0
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)
            reg_loss += config.l1_regularization * l1_loss
        
        if config.l2_regularization > 0:
            l2_loss = 0.0
            for param in model.parameters():
                l2_loss += torch.norm(param, 2)
            reg_loss += config.l2_regularization * l2_loss
        
        return reg_loss
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, config: HyperparameterConfig) -> Tuple[float, List[float]]:
        """
        Train model for one epoch with gradient tracking.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            config: Hyperparameter configuration
            
        Returns:
            Tuple of (average_loss, gradient_norms)
        """
        model.train()
        total_loss = 0.0
        gradient_norms = []
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Compute primary loss
            if config.loss_function == 'kldiv':
                outputs = torch.log_softmax(outputs, dim=1)
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
            
            # Add regularization
            reg_loss = self.compute_regularization_loss(model, config)
            total_loss_with_reg = loss + reg_loss
            
            total_loss_with_reg.backward()
            
            # Track gradient norms
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            gradient_norms.append(total_grad_norm)
            
            # Apply gradient clipping if enabled
            if config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader), gradient_norms
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module, config: HyperparameterConfig) -> float:
        """
        Validate model for one epoch.
        
        Args:
            model: PyTorch model
            val_loader: Validation data loader
            criterion: Loss function
            config: Hyperparameter configuration
            
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = model(sequences)
                
                if config.loss_function == 'kldiv':
                    outputs = torch.log_softmax(outputs, dim=1)
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def evaluate_config(self, config_dict: Dict[str, Any], max_epochs: int = 30, use_cv: bool = False) -> TrialResult:
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            config_dict: Dictionary of hyperparameter values
            max_epochs: Maximum number of training epochs
            use_cv: Whether to use cross-validation
            
        Returns:
            TrialResult with evaluation metrics
        """
        start_time = time.time()
        
        # Validate and create config
        config_dict = validate_config_compatibility(config_dict)
        config = HyperparameterConfig.from_dict(config_dict)
        
        # Create model
        model = self.model_class(
            num_blocks=config.depth,
            base_channels=config.base_channels,
            dropout=config.dropout,
            num_classes=config.num_classes
        ).to(self.device)
        
        # Create training components
        optimizer = self.create_optimizer(model, config)
        scheduler = self.create_scheduler(optimizer, config)
        criterion = self.create_criterion(config)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Training loop
        train_losses = []
        val_losses = []
        all_gradient_norms = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            train_loss, grad_norms = self.train_epoch(model, train_loader, criterion, optimizer, config)
            val_loss = self.validate_epoch(model, val_loader, criterion, config)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            all_gradient_norms.extend(grad_norms)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    break
            
            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Cross-validation if requested
        cv_scores = []
        if use_cv:
            cv_scores = self.cross_validate_config(config, k_folds=3, max_epochs=max_epochs // 2)
        
        duration = time.time() - start_time
        final_lr = optimizer.param_groups[0]['lr']
        
        return TrialResult(
            config=config_dict,
            val_loss=best_val_loss,
            val_losses=val_losses,
            train_losses=train_losses,
            params=sum(p.numel() for p in model.parameters()),
            duration_s=duration,
            epochs_trained=epoch + 1,
            final_lr=final_lr,
            cv_scores=cv_scores,
            cv_mean=np.mean(cv_scores) if cv_scores else 0.0,
            cv_std=np.std(cv_scores) if cv_scores else 0.0,
            gradient_norms=all_gradient_norms[:100]  # Store first 100 for analysis
        )
    
    def cross_validate_config(self, config: HyperparameterConfig, k_folds: int = 3, max_epochs: int = 15) -> List[float]:
        """
        Perform k-fold cross-validation for a configuration.
        
        Args:
            config: Hyperparameter configuration
            k_folds: Number of folds
            max_epochs: Maximum epochs per fold
            
        Returns:
            List of validation scores for each fold
        """
        # Combine train and val datasets for CV
        combined_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        indices = list(range(len(combined_dataset)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            # Create fold datasets
            train_subset = torch.utils.data.Subset(combined_dataset, train_idx)
            val_subset = torch.utils.data.Subset(combined_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
            
            # Create model and training components
            model = self.model_class(
                num_blocks=config.depth,
                base_channels=config.base_channels,
                dropout=config.dropout,
                num_classes=config.num_classes
            ).to(self.device)
            
            optimizer = self.create_optimizer(model, config)
            scheduler = self.create_scheduler(optimizer, config)
            criterion = self.create_criterion(config)
            
            # Training loop for this fold
            best_val_loss = float('inf')
            for epoch in range(max_epochs):
                train_loss, _ = self.train_epoch(model, train_loader, criterion, optimizer, config)
                val_loss = self.validate_epoch(model, val_loader, criterion, config)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
            
            cv_scores.append(best_val_loss)
        
        return cv_scores
    
    def random_search(self, n_trials: int = 50, search_space_type: SearchSpaceType = SearchSpaceType.COMPREHENSIVE, 
                     max_epochs: int = 30, use_cv: bool = False) -> List[TrialResult]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            n_trials: Number of random trials
            search_space_type: Type of search space to use
            max_epochs: Maximum epochs per trial
            use_cv: Whether to use cross-validation
            
        Returns:
            List of trial results
        """
        print(f"🔍 Starting random search with {n_trials} trials...")
        search_space = get_search_space(search_space_type)
        
        results = []
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            config_dict = sample_random_config(search_space)
            result = self.evaluate_config(config_dict, max_epochs, use_cv)
            results.append(result)
            
            print(f"  Val loss: {result.val_loss:.4f}, Duration: {result.duration_s:.1f}s")
        
        self.results.extend(results)
        return results
    
    def grid_search(self, search_space_type: SearchSpaceType = SearchSpaceType.BASIC, 
                   max_epochs: int = 20, use_cv: bool = False, max_combinations: int = 100) -> List[TrialResult]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            search_space_type: Type of search space to use
            max_epochs: Maximum epochs per trial
            use_cv: Whether to use cross-validation
            max_combinations: Maximum number of combinations to try
            
        Returns:
            List of trial results
        """
        search_space = get_search_space(search_space_type)
        
        # Generate all combinations
        keys, values = zip(*search_space.items())
        combinations = list(itertools.product(*values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            print(f"⚠️  Too many combinations ({len(combinations)}), sampling {max_combinations}")
            combinations = random.sample(combinations, max_combinations)
        
        print(f"🔍 Starting grid search with {len(combinations)} combinations...")
        
        results = []
        for i, combination in enumerate(combinations):
            print(f"Combination {i + 1}/{len(combinations)}")
            
            config_dict = dict(zip(keys, combination))
            result = self.evaluate_config(config_dict, max_epochs, use_cv)
            results.append(result)
            
            print(f"  Val loss: {result.val_loss:.4f}, Duration: {result.duration_s:.1f}s")
        
        self.results.extend(results)
        return results
    
    def get_best_result(self, results: List[TrialResult] = None) -> TrialResult:
        """
        Get the best result from trials.
        
        Args:
            results: List of results to search (uses self.results if None)
            
        Returns:
            Best trial result
        """
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results available")
        
        # Use CV mean if available, otherwise validation loss
        best_result = min(results, key=lambda r: r.cv_mean if r.cv_scores else r.val_loss)
        return best_result
    
    def save_results(self, filename: str = None) -> None:
        """
        Save tuning results to file.
        
        Args:
            filename: Optional filename override
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_results_{timestamp}.pkl"
        
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Saved {len(self.results)} results to {filepath}")
    
    def load_results(self, filename: str) -> None:
        """
        Load tuning results from file.
        
        Args:
            filename: Filename to load from
        """
        results_dir = Path(__file__).parent / 'results'
        filepath = results_dir / filename
        
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        
        print(f"Loaded {len(self.results)} results from {filepath}")
    
    def optimize_and_save_best(self, strategy: OptimizationStrategy = OptimizationStrategy.RANDOM,
                              search_space_type: SearchSpaceType = SearchSpaceType.COMPREHENSIVE,
                              n_trials: int = 50, model_name: str = 'promoter_cnn') -> HyperparameterConfig:
        """
        Run optimization and save the best configuration.
        
        Args:
            strategy: Optimization strategy to use
            search_space_type: Type of search space
            n_trials: Number of trials (for random search)
            model_name: Name of the model for saving config
            
        Returns:
            Best hyperparameter configuration
        """
        print(f"🚀 Starting {strategy.value} optimization...")
        
        if strategy == OptimizationStrategy.RANDOM:
            results = self.random_search(n_trials, search_space_type, use_cv=True)
        elif strategy == OptimizationStrategy.GRID:
            results = self.grid_search(search_space_type, use_cv=True)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented yet")
        
        # Get and save best result
        best_result = self.get_best_result(results)
        best_config = HyperparameterConfig.from_dict(best_result.config)
        best_config.model_name = model_name
        
        save_config(best_config, model_name)
        
        print(f"✅ Best configuration saved!")
        print(f"   Validation loss: {best_result.val_loss:.4f}")
        if best_result.cv_scores:
            print(f"   CV mean: {best_result.cv_mean:.4f} ± {best_result.cv_std:.4f}")
        
        # Save detailed results
        self.save_results()
        
        return best_config
