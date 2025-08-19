#!/usr/bin/env python3
"""
Unified Hyperparameter Tuning for CNN Models

This script provides a simplified, unified hyperparameter tuning system that:
- Works with both CUDA and MPS (Apple Silicon) devices
- Removes redundancies and overlaps from the existing tuning code
- Provides a clean, easy-to-use interface for hyperparameter optimization
- Supports multiple search strategies (random search, grid search)
- Automatically saves best configurations

Usage:
    python hyperparameter_tuning.py --trials 50 --strategy random
    python hyperparameter_tuning.py --trials 20 --strategy grid --search-space quick
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


# ================================
# Configuration and Results Classes
# ================================

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning"""
    # Model architecture
    depth: int = 2
    base_channels: int = 32
    dropout: float = 0.3
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    
    # Scheduler
    scheduler: str = 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Loss function
    loss_function: str = 'kldiv'
    
    # Training control
    max_epochs: int = 30
    early_stopping_patience: int = 8


@dataclass
class TuningResult:
    """Results from a single hyperparameter trial"""
    config: Dict[str, Any]
    val_loss: float
    train_loss: float
    params: int
    duration_s: float
    epochs_trained: int
    final_lr: float
    converged: bool = True


# ================================
# Device Management
# ================================

class DeviceManager:
    """Device manager that always prefers CUDA for optimal performance"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device, self.device_name = self._select_device()
        
        if verbose:
            print(f"🖥️  Using device: {self.device_name} ({self.device})")
            if self.device_name == "cuda":
                print(f"   GPU: {torch.cuda.get_device_name()}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   Memory: {total_memory:.1f}GB")
                print(f"   Compute Capability: {torch.cuda.get_device_capability()}")
                print(f"   CUDA Version: {torch.version.cuda}")
                # Optimize batch sizes based on GPU memory
                if total_memory >= 8.0:
                    print(f"   Recommended batch sizes: 64-128 (high memory GPU)")
                elif total_memory >= 4.0:
                    print(f"   Recommended batch sizes: 32-64 (medium memory GPU)")
                else:
                    print(f"   Recommended batch sizes: 16-32 (low memory GPU)")
    
    def _select_device(self) -> Tuple[torch.device, str]:
        """Select the best available device (always prefer CUDA)"""
        if torch.cuda.is_available():
            # Test CUDA functionality
            try:
                torch.cuda.empty_cache()  # Clear any existing cache
                test_tensor = torch.randn(2, 3, device='cuda')
                test_result = test_tensor * 2.0
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"✅ CUDA available and functional")
                return torch.device("cuda"), "cuda"
            except Exception as e:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"⚠️  CUDA available but not functional, falling back: {e}")
        
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Test MPS compatibility with a simple operation
            try:
                test_tensor = torch.randn(2, 3, dtype=torch.float32).to("mps")
                test_result = test_tensor * 2.0
                return torch.device("mps"), "mps"
            except Exception as e:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"⚠️  MPS available but not compatible, falling back to CPU: {e}")
                return torch.device("cpu"), "cpu"
        
        return torch.device("cpu"), "cpu"
    
    def to_device(self, data):
        """Move data to device with compatibility handling"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            if self.device_name == "cuda":
                # CUDA optimized transfer with non-blocking
                return data.to(self.device, non_blocking=True)
            elif self.device_name == "mps":
                # Ensure consistent tensor type for MPS compatibility
                return data.to(self.device, dtype=torch.float32)
            else:
                return data.to(self.device)
        else:
            return data
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get optimized DataLoader kwargs for the current device"""
        if self.device_name == "cuda":
            # CUDA optimizations: pin memory and multiple workers for maximum throughput
            import multiprocessing
            num_workers = min(8, max(2, multiprocessing.cpu_count() // 2))  # Use more workers
            return {
                "pin_memory": True, 
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 4,  # Increased prefetch for better GPU utilization
                "drop_last": True,  # Ensure consistent batch sizes for cuDNN optimization
                "pin_memory_device": "cuda"  # Pin to specific CUDA device
            }
        else:
            return {"num_workers": 0, "drop_last": False}  # MPS/CPU: single worker for stability
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current device memory information"""
        if self.device_name == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
                "utilization": (reserved / total) * 100
            }
        else:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0, "utilization": 0.0}
    
    def clear_cache(self):
        """Clear device cache to free memory"""
        if self.device_name == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                memory_info = self.get_memory_info()
                print(f"🧹 CUDA cache cleared - Free: {memory_info['free_gb']:.1f}GB")
    
    def find_optimal_batch_size(self, model, sample_input_shape, max_batch_size=1024):
        """Find the largest batch size that fits in GPU memory"""
        if self.device_name != "cuda":
            return 32  # Default for non-CUDA devices
        
        print("🔍 Finding optimal batch size for maximum GPU utilization...")
        
        # Create a dummy model for testing
        test_model = model.to(self.device)
        test_model.train()
        
        optimal_batch_size = 32
        
        for batch_size in [32, 64, 128, 256, 512, 1024, 2048]:
            if batch_size > max_batch_size:
                break
                
            try:
                self.clear_cache()
                
                # Create test batch
                test_input = torch.randn(batch_size, *sample_input_shape[1:], device=self.device)
                test_target = torch.randn(batch_size, 5, device=self.device).softmax(dim=1)
                
                # Test forward pass
                with torch.cuda.amp.autocast():
                    output = test_model(test_input)
                    loss = nn.KLDivLoss(reduction='batchmean')(
                        torch.log_softmax(output, dim=1), test_target
                    )
                
                # Test backward pass
                loss.backward()
                
                # Check memory usage
                memory_info = self.get_memory_info()
                if memory_info['utilization'] < 90:  # Keep some headroom
                    optimal_batch_size = batch_size
                    print(f"   ✅ Batch size {batch_size}: {memory_info['utilization']:.1f}% GPU memory")
                else:
                    print(f"   ⚠️  Batch size {batch_size}: {memory_info['utilization']:.1f}% GPU memory (too high)")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ Batch size {batch_size}: Out of memory")
                    break
                else:
                    raise e
        
        print(f"🎯 Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size


# ================================
# Dataset and Model
# ================================

class PromoterDataset(Dataset):
    """Dataset for promoter sequences with 5-component probability targets"""
    
    def __init__(self, sequences: List[str], targets: np.ndarray, max_length: int = 600):
        self.sequences = sequences
        self.targets = targets
        self.max_length = max_length
        self.dna_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    
    def __len__(self):
        return len(self.sequences)
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode DNA sequence to one-hot representation"""
        # Truncate or pad sequence
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + 'N' * (self.max_length - len(sequence))
        
        # Convert to numerical encoding
        encoded = np.array([self.dna_dict.get(base.upper(), 4) for base in sequence])
        
        # One-hot encode
        one_hot = np.zeros((self.max_length, 5))
        one_hot[np.arange(self.max_length), encoded] = 1
        
        return one_hot.T  # Shape: (5, max_length) for Conv1d
    
    def __getitem__(self, idx):
        sequence = self.encode_sequence(self.sequences[idx])
        target = self.targets[idx].astype(np.float32)
        
        # Normalize target to probabilities
        total = float(np.sum(target))
        if total <= 0:
            target = np.ones_like(target, dtype=np.float32) / target.shape[0]
        else:
            target = target / total
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class PromoterCNN(nn.Module):
    """Simplified CNN for promoter sequence classification"""
    
    def __init__(self, sequence_length: int = 600, num_blocks: int = 2, 
                 base_channels: int = 32, dropout: float = 0.3, num_classes: int = 5):
        super().__init__()
        
        conv_layers = []
        in_ch = 5  # One-hot encoded DNA
        out_ch = base_channels
        
        # Build convolutional blocks
        for i in range(num_blocks):
            kernel_size = 11 if i == 0 else 7
            padding = 5 if i == 0 else 3
            
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Add pooling for first two blocks
            if i < min(2, num_blocks):
                conv_layers.append(nn.MaxPool1d(kernel_size=4))
            
            in_ch = out_ch
            out_ch = min(out_ch * 2, base_channels * 4)  # Cap channel growth
        
        # Global average pooling
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, max(32, in_ch // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, in_ch // 2), num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ================================
# Search Spaces
# ================================

def get_optimized_search_space(device_manager):
    """Get search space optimized for the current device"""
    base_space = {
        'depth': [1, 2, 3, 4],
        'base_channels': [16, 24, 32, 48, 64],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        'optimizer': ['adam', 'adamw', 'sgd'],
        'scheduler': ['plateau', 'cosine', 'step', 'none'],
        'scheduler_patience': [3, 5, 8, 10],
        'scheduler_factor': [0.3, 0.5, 0.7],
        'loss_function': ['kldiv', 'mse'],
        'gradient_accumulation_steps': [1, 2, 4]  # For effective larger batch sizes
    }
    
    # Optimize batch sizes based on device
    if device_manager.device_name == "cuda":
        memory_info = device_manager.get_memory_info()
        total_memory = memory_info.get('total_gb', 0)
        
        if total_memory >= 16.0:
            # High-end GPU: Use large batch sizes for maximum utilization
            base_space['batch_size'] = [128, 256, 512, 1024]
            base_space['effective_batch_size'] = [256, 512, 1024, 2048]
        elif total_memory >= 8.0:
            # Mid-range GPU: Use medium-large batch sizes
            base_space['batch_size'] = [64, 128, 256, 512]
            base_space['effective_batch_size'] = [128, 256, 512, 1024]
        elif total_memory >= 4.0:
            # Lower-end GPU: Use medium batch sizes
            base_space['batch_size'] = [32, 64, 128, 256]
            base_space['effective_batch_size'] = [64, 128, 256, 512]
        else:
            # Very limited GPU: Use smaller batch sizes
            base_space['batch_size'] = [16, 32, 64, 128]
            base_space['effective_batch_size'] = [32, 64, 128, 256]
    else:
        # CPU/MPS: Use smaller batch sizes
        base_space['batch_size'] = [16, 32, 64]
        base_space['effective_batch_size'] = [32, 64, 128]
    
    return base_space

# For backward compatibility
COMPREHENSIVE_SEARCH_SPACE = {
    'depth': [1, 2, 3, 4],
    'base_channels': [16, 24, 32, 48, 64],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_size': [32, 64, 128, 256],  # Increased default batch sizes
    'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'scheduler': ['plateau', 'cosine', 'step', 'none'],
    'scheduler_patience': [3, 5, 8, 10],
    'scheduler_factor': [0.3, 0.5, 0.7],
    'loss_function': ['kldiv', 'mse'],
    'gradient_accumulation_steps': [1, 2, 4]
}


# ================================
# Hyperparameter Tuner
# ================================

class HyperparameterTuner:
    """Unified hyperparameter tuning system"""
    
    def __init__(self, train_dataset, val_dataset, device_manager: DeviceManager):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device_manager = device_manager
        self.results = []
    
    def create_optimizer(self, model, config):
        """Create optimizer from configuration"""
        params = model.parameters()
        lr = config['learning_rate']
        wd = config.get('weight_decay', 1e-4)
        
        if config['optimizer'] == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif config['optimizer'] == 'sgd':
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            return optim.Adam(params, lr=lr, weight_decay=wd)
    
    def create_scheduler(self, optimizer, config):
        """Create learning rate scheduler from configuration"""
        scheduler_type = config.get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            patience = config.get('scheduler_patience', 5)
            factor = config.get('scheduler_factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        else:
            return None
    
    def create_criterion(self, config):
        """Create loss function from configuration"""
        loss_type = config.get('loss_function', 'kldiv')
        
        if loss_type == 'kldiv':
            return nn.KLDivLoss(reduction='batchmean')
        elif loss_type == 'mse':
            return nn.MSELoss()
        else:
            return nn.KLDivLoss(reduction='batchmean')
    
    def train_epoch(self, model, train_loader, criterion, optimizer, config):
        """Train model for one epoch with gradient accumulation support"""
        model.train()
        total_loss = 0.0
        accumulation_steps = max(1, config.get('gradient_accumulation_steps', 1))  # Ensure at least 1
        
        # CUDA optimizations
        if self.device_manager.device_name == "cuda":
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.device_manager.device_name == "cuda" else None
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = self.device_manager.to_device(sequences)
            targets = self.device_manager.to_device(targets)
            
            # Ensure consistent dtypes for MPS compatibility
            if self.device_manager.device_name == "mps":
                sequences = sequences.to(dtype=torch.float32)
                targets = targets.to(dtype=torch.float32)
            
            # Use autocast for CUDA mixed precision training
            if self.device_manager.device_name == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(sequences)
                    # Apply appropriate loss function
                    if isinstance(criterion, nn.KLDivLoss):
                        outputs = F.log_softmax(outputs, dim=1)
                    loss = criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
            else:
                outputs = model(sequences)
                # Apply appropriate loss function
                if isinstance(criterion, nn.KLDivLoss):
                    outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling for mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Perform optimizer step after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    # Gradient clipping with scaler
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
        
        # Handle any remaining gradients
        if accumulated_loss > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
        
        return total_loss / max(1, len(train_loader))
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate model for one epoch"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = self.device_manager.to_device(sequences)
                targets = self.device_manager.to_device(targets)
                
                # Ensure consistent dtypes for MPS compatibility
                if self.device_manager.device_name == "mps":
                    sequences = sequences.to(dtype=torch.float32)
                    targets = targets.to(dtype=torch.float32)
                
                outputs = model(sequences)
                
                # Apply appropriate loss function
                if isinstance(criterion, nn.KLDivLoss):
                    outputs = F.log_softmax(outputs, dim=1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / max(1, len(val_loader))
    
    def evaluate_config(self, config: Dict[str, Any]) -> TuningResult:
        """Evaluate a single hyperparameter configuration"""
        start_time = time.time()
        
        try:
            # Check for empty datasets
            if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
                raise ValueError("Empty dataset provided - cannot train model")
            
            # Create data loaders
            loader_kwargs = self.device_manager.get_dataloader_kwargs()
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=min(config['batch_size'], len(self.train_dataset)), 
                shuffle=True, 
                **loader_kwargs
            )
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=min(config['batch_size'], len(self.val_dataset)), 
                shuffle=False, 
                **loader_kwargs
            )
            
            # Additional safety check for empty loaders
            if len(train_loader) == 0 or len(val_loader) == 0:
                raise ValueError(f"Empty DataLoader created - batch_size={config['batch_size']}, "
                               f"train_size={len(self.train_dataset)}, val_size={len(self.val_dataset)}")
            
            # Create model
            model = PromoterCNN(
                num_blocks=config['depth'],
                base_channels=config['base_channels'],
                dropout=config['dropout']
            )
            # Move model to device and apply optimizations
            model = model.to(self.device_manager.device)
            if self.device_manager.device_name == "mps":
                model = model.to(dtype=torch.float32)
            elif self.device_manager.device_name == "cuda":
                # CUDA optimizations
                torch.cuda.empty_cache()  # Clear cache before training
                # Enable cuDNN optimizations
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Create training components
            optimizer = self.create_optimizer(model, config)
            scheduler = self.create_scheduler(optimizer, config)
            criterion = self.create_criterion(config)
            
            # Training loop
            best_val_loss = float('inf')
            best_train_loss = float('inf')
            patience_counter = 0
            epochs_trained = 0
            
            for epoch in range(config.get('max_epochs', 30)):
                # Train and validate
                train_loss = self.train_epoch(model, train_loader, criterion, optimizer, config)
                val_loss = self.validate_epoch(model, val_loader, criterion)
                
                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Track best losses
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                epochs_trained = epoch + 1
                
                # Early stopping
                if patience_counter >= config.get('early_stopping_patience', 8):
                    break
            
            duration = time.time() - start_time
            final_lr = optimizer.param_groups[0]['lr']
            
            # Clean up memory after training
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
                converged=patience_counter < config.get('early_stopping_patience', 8)
            )
            
        except Exception as e:
            # If MPS fails, try falling back to CPU
            if self.device_manager.device_name == "mps" and "MPSFloatType" in str(e):
                print(f"⚠️  MPS compatibility issue, falling back to CPU: {e}")
                try:
                    # Temporarily switch to CPU for this trial
                    original_device = self.device_manager.device
                    original_device_name = self.device_manager.device_name
                    self.device_manager.device = torch.device("cpu")
                    self.device_manager.device_name = "cpu"
                    
                    # Retry the evaluation
                    result = self.evaluate_config(config)
                    
                    # Restore original device
                    self.device_manager.device = original_device
                    self.device_manager.device_name = original_device_name
                    
                    return result
                except Exception as e2:
                    print(f"❌ Trial failed even on CPU: {e2}")
            else:
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
    
    def comprehensive_search(self, num_trials: int, seed: int = 42) -> List[TuningResult]:
        """Perform comprehensive hyperparameter optimization using random search"""
        print(f"🎯 Starting comprehensive hyperparameter optimization with {num_trials} trials...")
        random.seed(seed)
        
        # Get optimized search space for current device
        search_space = get_optimized_search_space(self.device_manager)
        
        # Show initial memory state and optimization info for CUDA
        if self.device_manager.device_name == "cuda":
            memory_info = self.device_manager.get_memory_info()
            print(f"🖥️  Initial CUDA memory: {memory_info['free_gb']:.1f}GB free / {memory_info['total_gb']:.1f}GB total")
            print(f"🚀 GPU Utilization Optimizations:")
            print(f"   - Batch sizes: {search_space['batch_size']}")
            print(f"   - Gradient accumulation: {search_space['gradient_accumulation_steps']}")
            print(f"   - Mixed precision training: Enabled")
            print(f"   - Optimized data loading: {self.device_manager.get_dataloader_kwargs()['num_workers']} workers")
        
        results = []
        for trial in range(1, num_trials + 1):
            # Sample random configuration from optimized search space
            config = {key: random.choice(values) for key, values in search_space.items()}
            
            # Clean up incompatible parameters
            if config.get('scheduler') != 'plateau':
                config.pop('scheduler_patience', None)
                config.pop('scheduler_factor', None)
            
            # Calculate effective batch size for reporting
            effective_batch_size = config['batch_size'] * config.get('gradient_accumulation_steps', 1)
            
            # Clear CUDA cache before each trial
            if self.device_manager.device_name == "cuda":
                self.device_manager.clear_cache()
            
            # Evaluate configuration
            result = self.evaluate_config(config)
            results.append(result)
            
            # Progress update with memory info and effective batch size
            memory_str = ""
            if self.device_manager.device_name == "cuda":
                memory_info = self.device_manager.get_memory_info()
                memory_str = f", mem={memory_info['utilization']:.1f}%"
            
            print(f"Trial {trial:3d}/{num_trials}: "
                  f"val_loss={result.val_loss:.6f}, "
                  f"batch={config['batch_size']}x{config.get('gradient_accumulation_steps', 1)}={effective_batch_size}, "
                  f"params={result.params:,}, "
                  f"time={result.duration_s:.1f}s{memory_str}")
            
            # Best so far every 10 trials
            if trial % 10 == 0:
                valid_results = [r for r in results if r.val_loss != float('inf')]
                if valid_results:
                    best = min(valid_results, key=lambda r: r.val_loss)
                    print(f"   📊 Best so far: {best.val_loss:.6f}")
                    # Show memory status
                    if self.device_manager.device_name == "cuda":
                        memory_info = self.device_manager.get_memory_info()
                        print(f"   🖥️  CUDA memory: {memory_info['utilization']:.1f}% used, {memory_info['free_gb']:.1f}GB free")
        
        return results


# ================================
# Data Loading
# ================================

def load_data(file_path: str) -> Tuple[List[str], np.ndarray]:
    """Load and prepare data for training"""
    print(f"📂 Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Component probability columns
    prob_cols = ['Component_1_Probability', 'Component_2_Probability', 
                'Component_3_Probability', 'Component_4_Probability', 'Component_5_Probability']
    
    # Clean data
    print(f"Initial data shape: {df.shape}")
    df = df.dropna(subset=['ProSeq'] + prob_cols)
    print(f"After cleaning: {df.shape}")
    
    # Extract sequences and targets
    sequences = df['ProSeq'].tolist()
    targets = df[prob_cols].values
    
    # Validate sequences
    valid_sequences = []
    valid_targets = []
    
    for i, seq in enumerate(sequences):
        if isinstance(seq, str) and len(seq) > 0:
            valid_sequences.append(seq)
            valid_targets.append(targets[i])
    
    sequences = valid_sequences
    targets = np.array(valid_targets)
    
    print(f"Final dataset: {len(sequences)} samples")
    print(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.1f}")
    
    return sequences, targets


def create_datasets(sequences: List[str], targets: np.ndarray, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1, 
                   seed: int = 42) -> Tuple[PromoterDataset, PromoterDataset, PromoterDataset]:
    """Create train/validation/test datasets with safety checks"""
    
    # Input validation
    if len(sequences) == 0:
        raise ValueError("No sequences provided")
    if len(targets) == 0:
        raise ValueError("No targets provided")
    if len(sequences) != len(targets):
        raise ValueError(f"Sequence count ({len(sequences)}) doesn't match target count ({len(targets)})")
    if len(sequences) < 10:
        raise ValueError(f"Dataset too small ({len(sequences)} samples). Need at least 10 samples.")
    
    # Stratify by dominant component
    labels = np.argmax(targets, axis=1)
    
    try:
        # Train/test split
        train_seq, test_seq, train_targets, test_targets = train_test_split(
            sequences, targets, test_size=1-train_ratio, random_state=seed, stratify=labels
        )
        
        # Train/validation split
        train_labels = np.argmax(train_targets, axis=1)
        val_size = val_ratio / train_ratio
        train_seq, val_seq, train_targets, val_targets = train_test_split(
            train_seq, train_targets, test_size=val_size, random_state=seed, stratify=train_labels
        )
    except ValueError as e:
        print(f"⚠️  Stratified split failed: {e}")
        print("Falling back to random split...")
        # Fallback to random split if stratified fails
        train_seq, test_seq, train_targets, test_targets = train_test_split(
            sequences, targets, test_size=1-train_ratio, random_state=seed
        )
        val_size = val_ratio / train_ratio
        train_seq, val_seq, train_targets, val_targets = train_test_split(
            train_seq, train_targets, test_size=val_size, random_state=seed
        )
    
    # Final validation
    if len(train_seq) == 0 or len(val_seq) == 0 or len(test_seq) == 0:
        raise ValueError("One of the dataset splits is empty after splitting")
    
    print(f"📊 Data splits: Train={len(train_seq)}, Val={len(val_seq)}, Test={len(test_seq)}")
    
    # Create datasets
    train_dataset = PromoterDataset(train_seq, train_targets)
    val_dataset = PromoterDataset(val_seq, val_targets)
    test_dataset = PromoterDataset(test_seq, test_targets)
    
    return train_dataset, val_dataset, test_dataset


# ================================
# Results Analysis and Saving
# ================================

def analyze_results(results: List[TuningResult], top_k: int = 5) -> List[TuningResult]:
    """Analyze and display tuning results"""
    valid_results = [r for r in results if r.val_loss != float('inf')]
    valid_results.sort(key=lambda r: r.val_loss)
    
    print(f"\n📊 HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    print(f"Total trials: {len(results)}")
    print(f"Successful trials: {len(valid_results)}")
    print(f"Failed trials: {len(results) - len(valid_results)}")
    
    if not valid_results:
        print("❌ No successful trials!")
        return []
    
    print(f"\n🏆 TOP {min(top_k, len(valid_results))} CONFIGURATIONS:")
    print("-" * 60)
    
    for i, result in enumerate(valid_results[:top_k]):
        print(f"\nRank {i+1}:")
        print(f"  Validation Loss: {result.val_loss:.6f}")
        print(f"  Train Loss: {result.train_loss:.6f}")
        print(f"  Parameters: {result.params:,}")
        print(f"  Training Time: {result.duration_s:.1f}s")
        print(f"  Epochs: {result.epochs_trained}")
        print(f"  Final LR: {result.final_lr:.2e}")
        print(f"  Converged: {result.converged}")
        
        # Show key hyperparameters
        config = result.config
        print(f"  Config:")
        for key in ['depth', 'base_channels', 'dropout', 'batch_size', 'learning_rate', 'optimizer']:
            if key in config:
                print(f"    {key}: {config[key]}")
    
    return valid_results[:top_k]


def save_results(results: List[TuningResult], best_config: Dict[str, Any], 
                save_dir: str = "tuning_results") -> None:
    """Save tuning results and best configuration"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_path / f"tuning_results_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save best configuration as JSON
    config_file = save_path / "best_hyperparameters.json"
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"💾 Results saved:")
    print(f"   Detailed results: {results_file}")
    print(f"   Best config: {config_file}")


def plot_results(results: List[TuningResult], save_path: Optional[str] = None) -> None:
    """Create visualization of tuning results"""
    valid_results = [r for r in results if r.val_loss != float('inf')]
    
    if len(valid_results) < 2:
        print("⚠️  Not enough valid results for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Validation loss distribution
    val_losses = [r.val_loss for r in valid_results]
    axes[0, 0].hist(val_losses, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Validation Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Validation Losses')
    
    # 2. Parameters vs Performance
    params = [r.params for r in valid_results]
    axes[0, 1].scatter(params, val_losses, alpha=0.6)
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Parameters vs Performance')
    
    # 3. Training time vs Performance
    durations = [r.duration_s for r in valid_results]
    axes[1, 0].scatter(durations, val_losses, alpha=0.6)
    axes[1, 0].set_xlabel('Training Time (s)')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Training Time vs Performance')
    
    # 4. Learning rate analysis
    lrs = [r.config.get('learning_rate', 0) for r in valid_results]
    axes[1, 1].scatter(lrs, val_losses, alpha=0.6)
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title('Learning Rate vs Performance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hyperparameter Tuning for CNN Models")
    parser.add_argument('--data', type=str, 
                       default='data/processed/ProSeq_with_5component_analysis.csv',
                       help='Path to the data file')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='tuning_results',
                       help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate result plots')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Trials: {args.trials}")
    print(f"Seed: {args.seed}")
    print("Strategy: Comprehensive Random Search")
    print("Device: Always prefer CUDA for optimal performance")
    
    # Initialize device manager (always prefer CUDA)
    device_manager = DeviceManager()
    
    # Load and prepare data
    sequences, targets = load_data(args.data)
    train_dataset, val_dataset, test_dataset = create_datasets(sequences, targets, seed=args.seed)
    
    # Initialize tuner
    tuner = HyperparameterTuner(train_dataset, val_dataset, device_manager)
    
    # Run comprehensive optimization
    print(f"\n🎯 Starting comprehensive hyperparameter optimization...")
    results = tuner.comprehensive_search(args.trials, args.seed)
    
    # Analyze results
    best_configs = analyze_results(results, top_k=5)
    
    if best_configs:
        best_config = best_configs[0].config
        
        # Save results
        save_results(results, best_config, args.save_dir)
        
        # Generate plots
        if args.plot:
            plot_path = Path(args.save_dir) / "comprehensive_tuning_analysis.png"
            plot_results(results, str(plot_path))
        
        print(f"\n✅ Comprehensive hyperparameter tuning completed!")
        print(f"   Best validation loss: {best_configs[0].val_loss:.6f}")
        print(f"   Best configuration saved to: {args.save_dir}/best_hyperparameters.json")
        
    else:
        print("❌ No successful configurations found!")


if __name__ == "__main__":
    main()
