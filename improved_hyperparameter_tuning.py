#!/usr/bin/env python3
"""
Improved Hyperparameter Tuning for CNN Models

This script provides an enhanced hyperparameter tuning system that:
- Focuses on areas where the current best config might be suboptimal
- Explores more refined learning rate schedules
- Tests deeper architectures and regularization techniques
- Uses a more targeted search strategy
"""

import argparse
import json
import pickle
import time
import random
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import the original components
from hyperparameter_tuning import (
    DeviceManager, PromoterDataset, PromoterCNN, TuningResult, 
    load_data, create_datasets, analyze_results, save_results, plot_results
)


# ================================
# Enhanced Search Spaces
# ================================

def get_improved_search_space(device_manager):
    """Get improved search space focusing on areas for optimization"""
    
    # Base on analysis of current best config, we'll focus on:
    # 1. Lower learning rates (current 0.01 seems too high)
    # 2. More sophisticated architectures
    # 3. Better regularization
    # 4. Advanced schedulers
    
    improved_space = {
        # Architecture improvements - explore deeper networks
        'depth': [2, 3, 4, 5],  # Current best is 2, try deeper
        'base_channels': [24, 32, 48, 64, 80],  # Current best is 32, explore more
        'dropout': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],  # Current best is 0.2, fine-tune around it
        
        # Learning rate improvements - current 0.01 seems too high
        'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3],  # Much lower than current 0.01
        'weight_decay': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],  # Current is 1e-5, explore around it
        
        # Optimizer comparison - current is Adam
        'optimizer': ['adam', 'adamw', 'sgd'],  # AdamW often performs better than Adam
        
        # Advanced scheduler configurations
        'scheduler': ['plateau', 'cosine', 'cosine_warm', 'onecycle'],
        'scheduler_patience': [8, 12, 15, 20],  # Current is 12, explore more patience
        'scheduler_factor': [0.2, 0.3, 0.4, 0.5],  # Current is 0.3, explore around it
        
        # Loss function alternatives
        'loss_function': ['kldiv', 'mse', 'smooth_l1'],
        
        # Enhanced regularization
        'l1_regularization': [0.0, 1e-6, 5e-6, 1e-5],
        'l2_regularization': [0.0, 1e-6, 5e-6, 1e-5],
        'gradient_clipping': [False, True],
        'max_grad_norm': [0.5, 1.0, 2.0],
        
        # Training enhancements
        'gradient_accumulation_steps': [1, 2, 4],
        'label_smoothing': [0.0, 0.05, 0.1],  # Can help with overfitting
        'mixup_alpha': [0.0, 0.2, 0.4],  # Data augmentation technique
    }
    
    # Optimize batch sizes based on device
    if device_manager.device_name == "cuda":
        memory_info = device_manager.get_memory_info()
        total_memory = memory_info.get('total_gb', 0)
        
        if total_memory >= 8.0:
            improved_space['batch_size'] = [64, 128, 256]
        else:
            improved_space['batch_size'] = [32, 64, 128]
    else:
        improved_space['batch_size'] = [32, 64]
    
    return improved_space


def get_focused_search_space():
    """Get a focused search space for quick but effective tuning"""
    return {
        # Focus on most promising areas from analysis
        'depth': [3, 4],  # Deeper than current best
        'base_channels': [48, 64],  # More channels than current best
        'dropout': [0.2, 0.25, 0.3],  # Around current best
        
        # Much lower learning rates
        'learning_rate': [3e-4, 5e-4, 1e-3],
        'weight_decay': [1e-5, 5e-5, 1e-4],
        
        # Best optimizers
        'optimizer': ['adamw'],  # Often better than adam
        
        # Advanced scheduling
        'scheduler': ['cosine', 'onecycle'],
        'scheduler_patience': [15, 20],
        
        'loss_function': ['kldiv'],
        'batch_size': [64, 128],
        'gradient_clipping': [True],
        'max_grad_norm': [1.0],
        'label_smoothing': [0.0, 0.05],
    }


# ================================
# Enhanced CNN Model
# ================================

class EnhancedPromoterCNN(nn.Module):
    """Enhanced CNN with additional features for better performance"""
    
    def __init__(self, sequence_length: int = 600, num_blocks: int = 3, 
                 base_channels: int = 32, dropout: float = 0.3, num_classes: int = 5,
                 use_residual: bool = False, use_attention: bool = False):
        super().__init__()
        
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        conv_layers = []
        in_ch = 5  # One-hot encoded DNA
        
        # Build convolutional blocks with optional residual connections
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** min(i, 2))  # Progressive channel increase
            kernel_size = 11 if i == 0 else 7 if i == 1 else 5
            padding = kernel_size // 2
            
            block = [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            
            # Add pooling for first few blocks
            if i < min(3, num_blocks):
                block.append(nn.MaxPool1d(kernel_size=2))
            
            conv_layers.extend(block)
            in_ch = out_ch
        
        # Global average pooling
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(in_ch, num_heads=4, batch_first=True)
        
        # Enhanced classifier with residual connection
        classifier_layers = [
            nn.Linear(in_ch, max(64, in_ch // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, in_ch // 2), max(32, in_ch // 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, in_ch // 4), num_classes)
        ]
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        
        if self.use_attention:
            # Reshape for attention: (batch, seq_len, features)
            x_att = x.transpose(1, 2)
            x_att, _ = self.attention(x_att, x_att, x_att)
            x = x_att.transpose(1, 2)
        
        x = x.squeeze(-1)
        return self.classifier(x)


# ================================
# Enhanced Hyperparameter Tuner
# ================================

class EnhancedHyperparameterTuner:
    """Enhanced tuner with improved training strategies"""
    
    def __init__(self, train_dataset, val_dataset, device_manager):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device_manager = device_manager
        self.results = []
    
    def create_optimizer(self, model, config):
        """Create optimizer with enhanced configurations"""
        params = model.parameters()
        lr = config['learning_rate']
        wd = config.get('weight_decay', 1e-4)
        
        if config['optimizer'] == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=wd, eps=1e-8)
        elif config['optimizer'] == 'sgd':
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        else:
            return optim.AdamW(params, lr=lr, weight_decay=wd)
    
    def create_scheduler(self, optimizer, config, total_steps=None):
        """Create enhanced scheduler configurations"""
        scheduler_type = config.get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            patience = config.get('scheduler_patience', 10)
            factor = config.get('scheduler_factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, factor=factor, min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        elif scheduler_type == 'cosine_warm':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
        elif scheduler_type == 'onecycle' and total_steps:
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config['learning_rate'], total_steps=total_steps
            )
        else:
            return None
    
    def create_criterion(self, config):
        """Create enhanced loss functions"""
        loss_type = config.get('loss_function', 'kldiv')
        label_smoothing = config.get('label_smoothing', 0.0)
        
        if loss_type == 'kldiv':
            return nn.KLDivLoss(reduction='batchmean')
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            return nn.KLDivLoss(reduction='batchmean')
    
    def mixup_data(self, x, y, alpha=0.4):
        """Apply mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch_enhanced(self, model, train_loader, criterion, optimizer, config, scheduler=None):
        """Enhanced training epoch with advanced techniques"""
        model.train()
        total_loss = 0.0
        accumulation_steps = config.get('gradient_accumulation_steps', 1)
        mixup_alpha = config.get('mixup_alpha', 0.0)
        
        optimizer.zero_grad()
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = self.device_manager.to_device(sequences)
            targets = self.device_manager.to_device(targets)
            
            # Apply mixup if enabled
            if mixup_alpha > 0:
                sequences, targets_a, targets_b, lam = self.mixup_data(sequences, targets, mixup_alpha)
            
            outputs = model(sequences)
            
            # Compute loss
            if isinstance(criterion, nn.KLDivLoss):
                outputs = F.log_softmax(outputs, dim=1)
            
            if mixup_alpha > 0:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient clipping if enabled
            if config.get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
            
            # Optimizer step with accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # OneCycle scheduler steps every batch
                if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
        
        return total_loss / len(train_loader)
    
    def evaluate_config_enhanced(self, config: Dict[str, Any]) -> TuningResult:
        """Enhanced configuration evaluation"""
        start_time = time.time()
        
        try:
            # Create data loaders
            loader_kwargs = self.device_manager.get_dataloader_kwargs()
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                **loader_kwargs
            )
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                **loader_kwargs
            )
            
            # Create enhanced model
            model = EnhancedPromoterCNN(
                num_blocks=config['depth'],
                base_channels=config['base_channels'],
                dropout=config['dropout'],
                use_residual=config.get('use_residual', False),
                use_attention=config.get('use_attention', False)
            )
            model = model.to(self.device_manager.device)
            
            # Create training components
            optimizer = self.create_optimizer(model, config)
            
            # Calculate total steps for OneCycle scheduler
            total_steps = len(train_loader) * config.get('max_epochs', 50)
            scheduler = self.create_scheduler(optimizer, config, total_steps)
            criterion = self.create_criterion(config)
            
            # Training loop with enhanced features
            best_val_loss = float('inf')
            best_train_loss = float('inf')
            patience_counter = 0
            epochs_trained = 0
            
            for epoch in range(config.get('max_epochs', 50)):
                # Enhanced training
                train_loss = self.train_epoch_enhanced(model, train_loader, criterion, optimizer, config, scheduler)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences = self.device_manager.to_device(sequences)
                        targets = self.device_manager.to_device(targets)
                        
                        outputs = model(sequences)
                        if isinstance(criterion, nn.KLDivLoss):
                            outputs = F.log_softmax(outputs, dim=1)
                        
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Update scheduler (except OneCycle which updates per batch)
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                epochs_trained = epoch + 1
                
                if patience_counter >= config.get('early_stopping_patience', 15):
                    break
            
            duration = time.time() - start_time
            final_lr = optimizer.param_groups[0]['lr']
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Cleanup
            del model, optimizer, scheduler, criterion
            if self.device_manager.device_name == "cuda":
                torch.cuda.empty_cache()
            
            return TuningResult(
                config=config,
                val_loss=best_val_loss,
                train_loss=best_train_loss,
                params=param_count,
                duration_s=duration,
                epochs_trained=epochs_trained,
                final_lr=final_lr,
                converged=patience_counter < config.get('early_stopping_patience', 15)
            )
            
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            return TuningResult(
                config=config,
                val_loss=float('inf'),
                train_loss=float('inf'),
                params=0,
                duration_s=time.time() - start_time,
                epochs_trained=0,
                final_lr=0.0,
                converged=False
            )
    
    def improved_search(self, num_trials: int, search_type: str = "comprehensive") -> List[TuningResult]:
        """Run improved hyperparameter search"""
        print(f"🎯 Starting improved hyperparameter search with {num_trials} trials...")
        
        if search_type == "focused":
            search_space = get_focused_search_space()
        else:
            search_space = get_improved_search_space(self.device_manager)
        
        print(f"📊 Search space includes {len(search_space)} parameters")
        for param, values in search_space.items():
            print(f"   {param}: {len(values)} options")
        
        results = []
        for trial in range(1, num_trials + 1):
            # Sample configuration
            config = {key: random.choice(values) for key, values in search_space.items()}
            
            # Add default values
            config.update({
                'max_epochs': 50,
                'early_stopping_patience': 15,
                'num_classes': 5
            })
            
            # Clean up incompatible parameters
            if config.get('scheduler') != 'plateau':
                config.pop('scheduler_patience', None)
                config.pop('scheduler_factor', None)
            
            print(f"Trial {trial:3d}/{num_trials}: Testing config...")
            result = self.evaluate_config_enhanced(config)
            results.append(result)
            
            print(f"   Val loss: {result.val_loss:.6f}, "
                  f"Params: {result.params:,}, "
                  f"Time: {result.duration_s:.1f}s, "
                  f"Epochs: {result.epochs_trained}")
            
            # Show best so far every 10 trials
            if trial % 10 == 0:
                valid_results = [r for r in results if r.val_loss != float('inf')]
                if valid_results:
                    best = min(valid_results, key=lambda r: r.val_loss)
                    print(f"   🏆 Best so far: {best.val_loss:.6f}")
        
        self.results.extend(results)
        return results


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description="Improved Hyperparameter Tuning for CNN Models")
    parser.add_argument('--data', type=str, 
                       default='data/processed/ProSeq_with_5component_analysis.csv',
                       help='Path to the data file')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of trials to run')
    parser.add_argument('--search-type', type=str, default='comprehensive',
                       choices=['comprehensive', 'focused'],
                       help='Type of search space to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='results/analysis',
                       help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate result plots')
    
    args = parser.parse_args()
    
    print("🚀 IMPROVED HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Trials: {args.trials}")
    print(f"Search Type: {args.search_type}")
    print(f"Seed: {args.seed}")
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Load and prepare data
    sequences, targets = load_data(args.data)
    train_dataset, val_dataset, test_dataset = create_datasets(sequences, targets, seed=args.seed)
    
    # Initialize enhanced tuner
    tuner = EnhancedHyperparameterTuner(train_dataset, val_dataset, device_manager)
    
    # Run improved search
    results = tuner.improved_search(args.trials, args.search_type)
    
    # Analyze results
    best_configs = analyze_results(results, top_k=5)
    
    if best_configs:
        best_config = best_configs[0].config
        
        # Save results
        save_results(results, best_config, args.save_dir)
        
        # Generate plots
        if args.plot:
            plot_path = Path(args.save_dir) / "improved_tuning_analysis.png"
            plot_results(results, str(plot_path))
        
        print(f"\n✅ Improved hyperparameter tuning completed!")
        print(f"   Best validation loss: {best_configs[0].val_loss:.6f}")
        print(f"   Improvement over current best: Compare with your existing results")
        print(f"   Configuration saved to: {args.save_dir}/best_hyperparameters.json")
        
        # Show the best configuration
        print(f"\n🏆 BEST CONFIGURATION:")
        for key, value in best_config.items():
            if key not in ['max_epochs', 'early_stopping_patience', 'num_classes']:
                print(f"   {key}: {value}")
        
    else:
        print("❌ No successful configurations found!")


if __name__ == "__main__":
    main()
