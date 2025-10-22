"""
Hyperparameter optimization for embedding-based CNN models.

This module provides:
1. Optuna-based hyperparameter optimization
2. Grid search for systematic exploration
3. Analysis of hyperparameter importance
4. Best configuration recommendations
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import json
import os
from datetime import datetime

from embedding_cnn_models import (
    EmbeddingCNN, PatchingCNN, BinaryClassificationDataset,
    count_parameters
)


class HyperparameterOptimizer:
    """Hyperparameter optimization for embedding CNN models."""
    
    def __init__(self, data_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.data_path = data_path
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Load and prepare data for optimization."""
        print("Loading data for hyperparameter optimization...")
        
        # Load data
        data_binary = pd.read_csv(self.data_path)
        data_binary = data_binary[['binary_classification', 'ProSeq']]
        
        # Filter sequences
        sequence_lengths = data_binary['ProSeq'].str.len()
        data_filtered = data_binary[sequence_lengths >= 600].copy()
        
        # Split data
        train_data, test_data = train_test_split(
            data_filtered, test_size=0.2, random_state=42, 
            stratify=data_filtered['binary_classification']
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42, 
            stratify=train_data['binary_classification']
        )
        
        # Create datasets
        train_dataset = BinaryClassificationDataset(train_data)
        val_dataset = BinaryClassificationDataset(val_data)
        test_dataset = BinaryClassificationDataset(test_data)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    def objective_embedding_cnn(self, trial):
        """Objective function for EmbeddingCNN optimization."""
        
        # Suggest hyperparameters
        embedding_layers = trial.suggest_int('embedding_layers', 2, 4)
        embedding_dims = []
        
        # Progressive embedding dimensions
        start_dim = trial.suggest_int('start_embedding_dim', 64, 256)
        for i in range(embedding_layers):
            dim = max(16, start_dim // (2 ** i))
            embedding_dims.append(dim)
        
        conv_layers = trial.suggest_int('conv_layers', 2, 4)
        conv_filters = []
        start_filters = trial.suggest_int('start_conv_filters', 32, 128)
        
        for i in range(conv_layers):
            filters = start_filters * (2 ** i)
            conv_filters.append(min(filters, 512))  # Cap at 512
        
        use_2d_conv = trial.suggest_categorical('use_2d_conv', [True, False])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create model
        model = EmbeddingCNN(
            embedding_dims=embedding_dims,
            conv_filters=conv_filters,
            use_2d_conv=use_2d_conv,
            dropout=dropout
        )
        
        # Check parameter count (avoid too large models)
        param_count = count_parameters(model)
        if param_count > 1_000_000:  # 1M parameter limit
            raise optuna.exceptions.TrialPruned()
        
        # Train and evaluate
        val_accuracy = self._train_and_evaluate(model, lr, weight_decay, max_epochs=15)
        
        return val_accuracy
    
    def objective_patching_cnn(self, trial):
        """Objective function for PatchingCNN optimization."""
        
        # Suggest hyperparameters
        patch_size = trial.suggest_int('patch_size', 2, 5)
        patch_embed_dim = trial.suggest_int('patch_embed_dim', 12, 32)
        
        conv_layers = trial.suggest_int('conv_layers', 2, 4)
        conv_filters = []
        start_filters = trial.suggest_int('start_conv_filters', 16, 64)
        
        for i in range(conv_layers):
            filters = start_filters * (2 ** i)
            conv_filters.append(min(filters, 256))  # Cap at 256 for patching
        
        # Motif sizes
        min_motif = trial.suggest_int('min_motif_size', 4, 7)
        max_motif = trial.suggest_int('max_motif_size', 8, 12)
        motif_sizes = list(range(min_motif, max_motif + 1))
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create model
        model = PatchingCNN(
            patch_size=patch_size,
            patch_embed_dim=patch_embed_dim,
            conv_filters=conv_filters,
            motif_sizes=motif_sizes,
            dropout=dropout
        )
        
        # Check parameter count
        param_count = count_parameters(model)
        if param_count > 500_000:  # 500K parameter limit for patching
            raise optuna.exceptions.TrialPruned()
        
        # Train and evaluate
        val_accuracy = self._train_and_evaluate(model, lr, weight_decay, max_epochs=15)
        
        return val_accuracy
    
    def _train_and_evaluate(self, model, lr, weight_decay, max_epochs=15):
        """Train model and return validation accuracy."""
        model = model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    outputs = model(batch_x)
                    predicted = (outputs > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_accuracy = correct / total
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_val_accuracy
    
    def optimize_embedding_cnn(self, n_trials=50, study_name="embedding_cnn_optimization"):
        """Optimize EmbeddingCNN hyperparameters."""
        print(f"Starting EmbeddingCNN optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(self.objective_embedding_cnn, n_trials=n_trials)
        
        return study
    
    def optimize_patching_cnn(self, n_trials=50, study_name="patching_cnn_optimization"):
        """Optimize PatchingCNN hyperparameters."""
        print(f"Starting PatchingCNN optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(self.objective_patching_cnn, n_trials=n_trials)
        
        return study
    
    def analyze_study(self, study, model_type="embedding"):
        """Analyze optimization results."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION RESULTS - {model_type.upper()}")
        print(f"{'='*60}")
        
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial value: {study.best_value:.4f}")
        
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Parameter importance
        if len(study.trials) > 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                print(f"\nParameter importance:")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {imp:.4f}")
            except Exception as e:
                print(f"Could not calculate parameter importance: {e}")
        
        return study.best_params
    
    def plot_optimization_results(self, study, model_type="embedding"):
        """Plot optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Optimization history
        ax = axes[0, 0]
        trials = [t for t in study.trials if t.value is not None]
        values = [t.value for t in trials]
        ax.plot(values, 'b-', alpha=0.7)
        ax.set_title(f'{model_type.title()} CNN Optimization History')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance (if available)
        ax = axes[0, 1]
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            ax.barh(params, importances)
            ax.set_title('Parameter Importance')
            ax.set_xlabel('Importance')
        except:
            ax.text(0.5, 0.5, 'Parameter importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Importance')
        
        # Plot 3: Value distribution
        ax = axes[1, 0]
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Validation Accuracy Distribution')
        ax.set_xlabel('Validation Accuracy')
        ax.set_ylabel('Frequency')
        ax.axvline(study.best_value, color='red', linestyle='--', 
                  label=f'Best: {study.best_value:.4f}')
        ax.legend()
        
        # Plot 4: Best vs worst trials comparison
        ax = axes[1, 1]
        sorted_trials = sorted(trials, key=lambda x: x.value, reverse=True)
        best_trials = sorted_trials[:5]
        worst_trials = sorted_trials[-5:]
        
        best_values = [t.value for t in best_trials]
        worst_values = [t.value for t in worst_trials]
        
        ax.bar(range(5), best_values, alpha=0.7, label='Best 5 trials', color='green')
        ax.bar(range(5, 10), worst_values, alpha=0.7, label='Worst 5 trials', color='red')
        ax.set_title('Best vs Worst Trials')
        ax.set_xlabel('Trial Rank')
        ax.set_ylabel('Validation Accuracy')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, study, model_type, output_dir="optimization_results"):
        """Save optimization results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best parameters
        best_params_file = f"{output_dir}/{model_type}_best_params_{timestamp}.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        # Save all trials
        trials_file = f"{output_dir}/{model_type}_all_trials_{timestamp}.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
        return best_params_file, trials_file


def run_comprehensive_optimization(data_path: str, n_trials_per_model=30):
    """Run comprehensive hyperparameter optimization for both models."""
    
    optimizer = HyperparameterOptimizer(data_path)
    
    results = {}
    
    # Optimize EmbeddingCNN
    print("="*60)
    print("OPTIMIZING EMBEDDING CNN")
    print("="*60)
    
    embedding_study = optimizer.optimize_embedding_cnn(n_trials=n_trials_per_model)
    embedding_best = optimizer.analyze_study(embedding_study, "embedding")
    optimizer.plot_optimization_results(embedding_study, "embedding")
    
    results['embedding'] = {
        'study': embedding_study,
        'best_params': embedding_best,
        'best_value': embedding_study.best_value
    }
    
    # Optimize PatchingCNN
    print("\n" + "="*60)
    print("OPTIMIZING PATCHING CNN")
    print("="*60)
    
    patching_study = optimizer.optimize_patching_cnn(n_trials=n_trials_per_model)
    patching_best = optimizer.analyze_study(patching_study, "patching")
    optimizer.plot_optimization_results(patching_study, "patching")
    
    results['patching'] = {
        'study': patching_study,
        'best_params': patching_best,
        'best_value': patching_study.best_value
    }
    
    # Save results
    optimizer.save_results(embedding_study, "embedding_cnn")
    optimizer.save_results(patching_study, "patching_cnn")
    
    # Compare best models
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    print(f"Best EmbeddingCNN validation accuracy: {results['embedding']['best_value']:.4f}")
    print(f"Best PatchingCNN validation accuracy: {results['patching']['best_value']:.4f}")
    
    if results['embedding']['best_value'] > results['patching']['best_value']:
        print("🏆 EmbeddingCNN achieved better performance")
        winner = 'embedding'
    else:
        print("🏆 PatchingCNN achieved better performance")
        winner = 'patching'
    
    print(f"\nRecommended configuration ({winner}):")
    for key, value in results[winner]['best_params'].items():
        print(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    # Example usage
    data_path = "../../../data/processed/ProSeq_binary_classification.csv"
    
    if os.path.exists(data_path):
        results = run_comprehensive_optimization(data_path, n_trials_per_model=20)
    else:
        print(f"Data file not found: {data_path}")
        print("Please update the data path and run again.")
