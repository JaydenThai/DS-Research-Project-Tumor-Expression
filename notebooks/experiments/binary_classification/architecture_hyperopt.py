#!/usr/bin/env python3
"""
CNN Architecture Hyperparameter Optimization for Binary Classification
Optimizes number of convolutional layers, kernel sizes, and fully connected layers
Uses Optuna for Bayesian optimization
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import json
import time
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print("CNN ARCHITECTURE HYPERPARAMETER OPTIMIZATION")
print("="*70)
print("🎯 Optimizing: Conv layers, kernel sizes, FC layers")
print("⚡ Using Optuna Bayesian optimization")
print("-" * 70)

# Load and prepare data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
data_path = os.path.join(project_root, 'data', 'processed', 'ProSeq_binary_classification.csv')
print(f"Loading data from: {data_path}")
data_binary = pd.read_csv(data_path)
data_binary = data_binary[['binary_classification', 'ProSeq']]

# Filter sequences >= 600 bp
sequence_lengths = data_binary['ProSeq'].str.len()
data_filtered = data_binary[sequence_lengths >= 600].copy()
print(f"Dataset size: {len(data_filtered)}")

# Class distribution analysis
class_dist = data_filtered['binary_classification'].value_counts()
print(f"\nClass distribution:")
for class_val, count in class_dist.items():
    percentage = count / len(data_filtered) * 100
    print(f"  Class {class_val}: {count} ({percentage:.1f}%)")

# Calculate class weights
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(data_filtered['binary_classification']),
                                   y=data_filtered['binary_classification'])
print(f"Balanced class weights: {class_weights}")

class BinaryClassificationDataset(Dataset):
    """Dataset for binary classification with DNA sequences"""
    def __init__(self, data, target_length=600):
        self.data = data.reset_index(drop=True)
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3}
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['ProSeq']
        target = self.data.iloc[idx]['binary_classification']
        one_hot_sequence = self.one_hot_encode(sequence)
        return one_hot_sequence, target
    
    def one_hot_encode(self, sequence):
        sequence = sequence[:self.target_length]
        one_hot = np.zeros((self.target_length, 4), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.dna_dict:
                one_hot[i, self.dna_dict[nucleotide]] = 1.0
        return one_hot

# Data splitting
train_data, test_data = train_test_split(data_filtered, test_size=0.2, random_state=42, 
                                       stratify=data_filtered['binary_classification'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, 
                                      stratify=train_data['binary_classification'])

print(f"\nData splits:")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")  
print(f"  Test: {len(test_data)} samples")

# Create datasets
train_dataset = BinaryClassificationDataset(train_data)
val_dataset = BinaryClassificationDataset(val_data)
test_dataset = BinaryClassificationDataset(test_data)

class FlexibleCNN(nn.Module):
    """Flexible CNN architecture with variable number of layers"""
    def __init__(self, 
                 input_channels=4,
                 input_length=600,
                 num_conv_layers=3,
                 conv_channels=[32, 64, 128],
                 kernel_sizes=[7, 5, 3],
                 pool_sizes=[2, 2, 2],
                 num_fc_layers=2,
                 fc_sizes=[256, 128],
                 dropout_rate=0.3,
                 use_batch_norm=True,
                 activation='relu'):
        
        super(FlexibleCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = input_channels
        current_length = input_length
        
        for i in range(num_conv_layers):
            # Ensure we have enough channels and kernel sizes
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
            pool_size = pool_sizes[i] if i < len(pool_sizes) else pool_sizes[-1]
            
            # Convolutional layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
            
            # Batch normalization
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_channels))
            else:
                self.bn_layers.append(nn.Identity())
            
            # Pooling layer
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            
            in_channels = out_channels
            current_length = current_length // pool_size
        
        # Calculate size after convolutions
        self.conv_output_size = in_channels * current_length
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()
        
        fc_input_size = self.conv_output_size
        
        for i in range(num_fc_layers):
            fc_output_size = fc_sizes[i] if i < len(fc_sizes) else fc_sizes[-1]
            
            self.fc_layers.append(nn.Linear(fc_input_size, fc_output_size))
            
            if use_batch_norm:
                self.fc_bn_layers.append(nn.BatchNorm1d(fc_output_size))
            else:
                self.fc_bn_layers.append(nn.Identity())
            
            fc_input_size = fc_output_size
        
        # Output layer
        self.output_layer = nn.Linear(fc_input_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [batch, length, channels] -> [batch, channels, length]
        x = x.transpose(1, 2)
        
        # Convolutional layers
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.pool_layers[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for i in range(self.num_fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_bn_layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Output layer
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()

class ArchitectureOptimizer:
    """Optuna-based architecture optimizer"""
    def __init__(self, device, train_dataset, val_dataset, class_weights):
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_weights = class_weights
        self.train_targets = train_data['binary_classification'].values
        
    def create_balanced_loader(self, dataset, targets, batch_size, shuffle=True):
        if shuffle:
            sample_weights = np.array([self.class_weights[t] for t in targets])
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def objective(self, trial):
        """Optuna objective function"""
        
        print(f"\n🔍 Trial {trial.number} starting...")
        
        # Architecture hyperparameters
        num_conv_layers = trial.suggest_int('num_conv_layers', 2, 5)
        num_fc_layers = trial.suggest_int('num_fc_layers', 1, 4)
        
        print(f"  📐 Architecture: {num_conv_layers} conv layers, {num_fc_layers} FC layers")
        
        # Convolutional layer parameters
        conv_channels = []
        kernel_sizes = []
        pool_sizes = []
        
        for i in range(num_conv_layers):
            # Channel progression: generally increasing
            base_channels = trial.suggest_int(f'conv_base_channels', 16, 128)
            multiplier = trial.suggest_float(f'conv_multiplier_{i}', 1.0, 3.0)
            channels = int(base_channels * (multiplier ** i))
            channels = min(channels, 512)  # Cap at 512
            conv_channels.append(channels)
            
            # Kernel sizes
            kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 15, step=2)  # Odd numbers
            kernel_sizes.append(kernel_size)
            
            # Pool sizes
            pool_size = trial.suggest_int(f'pool_size_{i}', 2, 4)
            pool_sizes.append(pool_size)
        
        # Fully connected layer parameters
        fc_sizes = []
        for i in range(num_fc_layers):
            fc_size = trial.suggest_int(f'fc_size_{i}', 32, 512)
            fc_sizes.append(fc_size)
        
        # Other hyperparameters
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu'])
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        try:
            # Create model
            model = FlexibleCNN(
                num_conv_layers=num_conv_layers,
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                pool_sizes=pool_sizes,
                num_fc_layers=num_fc_layers,
                fc_sizes=fc_sizes,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation
            ).to(self.device)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  🔢 Model parameters: {param_count:,}")
            
            # Skip if too many parameters (memory constraint)
            if param_count > 1_000_000:  # 1M parameters max
                print(f"  ⚠️  Skipping: too many parameters ({param_count:,} > 10M)")
                return 0.0
            
            # Create optimizer
            if optimizer_type == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:  # sgd
                momentum = trial.suggest_float('momentum', 0.5, 0.99)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            
            # Create data loaders
            train_loader = self.create_balanced_loader(self.train_dataset, self.train_targets, batch_size)
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop
            criterion = nn.BCELoss()
            num_epochs = 15  # Quick training for hyperparameter search
            print(f"  🏋️  Training for {num_epochs} epochs...")
            
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_probs = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    outputs = model(batch_x)
                    val_probs.extend(outputs.cpu().numpy())
                    
                    predicted = (outputs > 0.5).float()
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds) if len(set(val_preds)) > 1 else 0.0
            
            try:
                val_auc = roc_auc_score(val_targets, val_probs)
            except:
                val_auc = 0.0
            
            # Composite score (prioritizing F1 and AUC)
            score = 0.5 * val_f1 + 0.3 * val_auc + 0.2 * val_accuracy
            
            # Penalty for large models (parameter efficiency)
            efficiency_penalty = param_count / 1_000_000  # Penalty per million parameters
            score = score - 0.01 * efficiency_penalty
            
            print(f"  📊 Results: F1={val_f1:.4f}, AUC={val_auc:.4f}, Acc={val_accuracy:.4f}")
            print(f"  🎯 Final score: {score:.4f}")
            
            # Log key metrics for this trial
            trial.set_user_attr('val_accuracy', val_accuracy)
            trial.set_user_attr('val_f1', val_f1)
            trial.set_user_attr('val_auc', val_auc)
            trial.set_user_attr('param_count', param_count)
            trial.set_user_attr('both_classes_predicted', len(set(val_preds)) > 1)
            
            return score
            
        except Exception as e:
            print(f"  ❌ Trial {trial.number} failed: {e}")
            return 0.0
    
    def optimize(self, n_trials=100, timeout=14400):
        """Run optimization"""
        print(f"\n🚀 Starting architecture optimization")
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout} seconds")
        print("-" * 50)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Progress callback
        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number < 5:
                print(f"\n📈 Progress Update - Trial {trial.number}")
                if len(study.trials) > 1:
                    best_value = study.best_value
                    best_trial = study.best_trial.number
                    print(f"  🏆 Current best: Trial {best_trial} with score {best_value:.4f}")
                    
                    # Show best architecture so far
                    best_params = study.best_params
                    print(f"  🏗️  Best architecture: {best_params.get('num_conv_layers', 'N/A')} conv, {best_params.get('num_fc_layers', 'N/A')} FC")
                print("-" * 50)
        
        # Optimize
        start_time = time.time()
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, callbacks=[progress_callback])
        optimization_time = time.time() - start_time
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"Time taken: {optimization_time/60:.2f} minutes")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Best score: {study.best_value:.4f}")
        
        best_params = study.best_params
        best_trial = study.best_trial
        
        print(f"\n📊 BEST ARCHITECTURE:")
        print(f"  Convolutional layers: {best_params['num_conv_layers']}")
        print(f"  Fully connected layers: {best_params['num_fc_layers']}")
        print(f"  Parameter count: {best_trial.user_attrs.get('param_count', 'N/A')}")
        print(f"  Validation F1: {best_trial.user_attrs.get('val_f1', 'N/A'):.4f}")
        print(f"  Validation AUC: {best_trial.user_attrs.get('val_auc', 'N/A'):.4f}")
        print(f"  Validation Accuracy: {best_trial.user_attrs.get('val_accuracy', 'N/A'):.4f}")
        
        return study
    
    def visualize_optimization(self, study, save_path):
        """Create visualization of optimization results"""
        print(f"\n📈 Creating optimization visualizations...")
        
        # Extract trial data
        trials_df = study.trials_dataframe()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('CNN Architecture Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Optimization history
        axes[0, 0].plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Number of conv layers vs performance
        if 'params_num_conv_layers' in trials_df.columns:
            conv_layers = trials_df['params_num_conv_layers'].dropna()
            values = trials_df.loc[conv_layers.index, 'value']
            axes[0, 1].scatter(conv_layers, values, alpha=0.6, c='red')
            axes[0, 1].set_xlabel('Number of Conv Layers')
            axes[0, 1].set_ylabel('Objective Value')
            axes[0, 1].set_title('Conv Layers vs Performance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Number of FC layers vs performance
        if 'params_num_fc_layers' in trials_df.columns:
            fc_layers = trials_df['params_num_fc_layers'].dropna()
            values = trials_df.loc[fc_layers.index, 'value']
            axes[0, 2].scatter(fc_layers, values, alpha=0.6, c='green')
            axes[0, 2].set_xlabel('Number of FC Layers')
            axes[0, 2].set_ylabel('Objective Value')
            axes[0, 2].set_title('FC Layers vs Performance')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Parameter count vs performance
        param_counts = [trial.user_attrs.get('param_count', 0) for trial in study.trials]
        values = [trial.value for trial in study.trials if trial.value is not None]
        if param_counts and values:
            axes[1, 0].scatter(param_counts[:len(values)], values, alpha=0.6, c='purple')
            axes[1, 0].set_xlabel('Parameter Count')
            axes[1, 0].set_ylabel('Objective Value')
            axes[1, 0].set_title('Model Size vs Performance')
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Dropout rate vs performance
        if 'params_dropout_rate' in trials_df.columns:
            dropout_rates = trials_df['params_dropout_rate'].dropna()
            values = trials_df.loc[dropout_rates.index, 'value']
            axes[1, 1].scatter(dropout_rates, values, alpha=0.6, c='orange')
            axes[1, 1].set_xlabel('Dropout Rate')
            axes[1, 1].set_ylabel('Objective Value')
            axes[1, 1].set_title('Dropout vs Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Learning rate vs performance
        if 'params_learning_rate' in trials_df.columns:
            learning_rates = trials_df['params_learning_rate'].dropna()
            values = trials_df.loc[learning_rates.index, 'value']
            axes[1, 2].scatter(learning_rates, values, alpha=0.6, c='cyan')
            axes[1, 2].set_xlabel('Learning Rate')
            axes[1, 2].set_ylabel('Objective Value')
            axes[1, 2].set_title('Learning Rate vs Performance')
            axes[1, 2].set_xscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Activation function performance
        if 'params_activation' in trials_df.columns:
            activation_perf = trials_df.groupby('params_activation')['value'].mean()
            axes[2, 0].bar(activation_perf.index, activation_perf.values)
            axes[2, 0].set_xlabel('Activation Function')
            axes[2, 0].set_ylabel('Average Objective Value')
            axes[2, 0].set_title('Activation Function Performance')
            axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Optimizer performance
        if 'params_optimizer' in trials_df.columns:
            optimizer_perf = trials_df.groupby('params_optimizer')['value'].mean()
            axes[2, 1].bar(optimizer_perf.index, optimizer_perf.values)
            axes[2, 1].set_xlabel('Optimizer')
            axes[2, 1].set_ylabel('Average Objective Value')
            axes[2, 1].set_title('Optimizer Performance')
        
        # 9. Top 10 trials summary
        top_trials = trials_df.nlargest(10, 'value')
        trial_numbers = top_trials['number']
        trial_values = top_trials['value']
        axes[2, 2].bar(range(len(trial_numbers)), trial_values)
        axes[2, 2].set_xlabel('Top Trials (by rank)')
        axes[2, 2].set_ylabel('Objective Value')
        axes[2, 2].set_title('Top 10 Trials')
        axes[2, 2].set_xticks(range(len(trial_numbers)))
        axes[2, 2].set_xticklabels([f'T{int(n)}' for n in trial_numbers], rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to '{save_path}'")

def evaluate_best_model(study, train_dataset, val_dataset, test_dataset, class_weights, device):
    """Evaluate the best model on test set"""
    print(f"\n🔬 EVALUATING BEST MODEL ON TEST SET")
    print("-" * 50)
    
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Reconstruct best model
    num_conv_layers = best_params['num_conv_layers']
    num_fc_layers = best_params['num_fc_layers']
    
    conv_channels = []
    kernel_sizes = []
    pool_sizes = []
    
    for i in range(num_conv_layers):
        base_channels = best_params['conv_base_channels']
        multiplier = best_params[f'conv_multiplier_{i}']
        channels = int(base_channels * (multiplier ** i))
        channels = min(channels, 512)
        conv_channels.append(channels)
        
        kernel_sizes.append(best_params[f'kernel_size_{i}'])
        pool_sizes.append(best_params[f'pool_size_{i}'])
    
    fc_sizes = []
    for i in range(num_fc_layers):
        fc_sizes.append(best_params[f'fc_size_{i}'])
    
    model = FlexibleCNN(
        num_conv_layers=num_conv_layers,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        num_fc_layers=num_fc_layers,
        fc_sizes=fc_sizes,
        dropout_rate=best_params['dropout_rate'],
        use_batch_norm=best_params['use_batch_norm'],
        activation=best_params['activation']
    ).to(device)
    
    # Create optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                             lr=best_params['learning_rate'], 
                             weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                              lr=best_params['learning_rate'], 
                              weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                            lr=best_params['learning_rate'], 
                            momentum=best_params.get('momentum', 0.9),
                            weight_decay=best_params['weight_decay'])
    
    # Train with more epochs for final evaluation
    train_targets = train_data['binary_classification'].values
    sample_weights = np.array([class_weights[t] for t in train_targets])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    criterion = nn.BCELoss()
    num_epochs = 50  # More thorough training for final model
    
    print(f"Training best model for {num_epochs} epochs...")
    
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.float().to(device)
                    
                    outputs = model(batch_x)
                    predicted = (outputs > 0.5).float()
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_f1 = f1_score(val_targets, val_preds) if len(set(val_preds)) > 1 else 0.0
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
            
            print(f"Epoch {epoch+1}: Val F1 = {val_f1:.4f}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x)
            test_probs.extend(outputs.cpu().numpy())
            
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    # Calculate final metrics
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds) if len(set(test_preds)) > 1 else 0.0
    test_auc = roc_auc_score(test_targets, test_probs) if len(set(test_targets)) > 1 else 0.0
    test_cm = confusion_matrix(test_targets, test_preds)
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC Score: {test_auc:.4f}")
    print(f"Both classes predicted: {len(set(test_preds)) > 1}")
    print(f"Confusion Matrix:\n{test_cm}")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    return {
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'confusion_matrix': test_cm.tolist(),
        'param_count': param_count,
        'both_classes_predicted': len(set(test_preds)) > 1
    }

# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create optimizer
    optimizer = ArchitectureOptimizer(device, train_dataset, val_dataset, class_weights)
    
    # Run optimization (start with fewer trials for testing)
    print(f"\n🚀 Starting CNN architecture optimization...")
    print(f"Device: {device}")
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    study = optimizer.optimize(n_trials=20, timeout=3600)  # 1 hour max, 20 trials for testing
    
    # Visualize results
    viz_path = os.path.join(project_root, 'results', 'cnn_architecture_optimization.png')
    optimizer.visualize_optimization(study, viz_path)
    
    # Evaluate best model
    final_results = evaluate_best_model(study, train_dataset, val_dataset, test_dataset, class_weights, device)
    
    # Save results
    results = {
        'study_summary': {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_time_minutes': sum(trial.duration.total_seconds() for trial in study.trials if trial.duration) / 60
        },
        'final_evaluation': final_results
    }
    
    results_path = os.path.join(project_root, 'results', 'cnn_architecture_optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to '{results_path}'")
    print(f"\n✅ CNN ARCHITECTURE OPTIMIZATION COMPLETE!")
    print(f"🏗️  Best architecture found with {final_results['param_count']:,} parameters")
    print(f"🎯 Final test F1 score: {final_results['test_f1']:.4f}")
