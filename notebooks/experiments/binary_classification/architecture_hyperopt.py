#!/usr/bin/env python3
"""
CNN Architecture Hyperparameter Optimization for Binary Classification (Extensive Tuning for Small Datasets)

Optimizes a wide range of hyperparameters using K-Fold Cross-Validation within each trial
for robust evaluation. Focuses on finding well-regularized models to prevent overfitting.
Saves all trial results to a comprehensive CSV file and generates extensive visualizations,
including per-trial K-Fold training curves (mean +/- std).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import json
import time
import random
import warnings
import sys
from io import StringIO
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Configuration ---
SEED = 1356294
EXPERIMENT_NAME = "binary_classification_cnn_extensive"
N_SPLITS_K_FOLD = 5 # Number of folds for cross-validation

# Set seeds for reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()

def get_timestamped_filename(base_name, extension):
    """Helper Function: Generate a timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

class OutputLogger:
    """Helper Function: Captures all print outputs for logging"""
    def __init__(self):
        self.log_buffer = StringIO(); self.original_stdout = sys.stdout; self.start_time = time.time()
    def start_logging(self): sys.stdout = self
    def stop_logging(self): sys.stdout = self.original_stdout
    def write(self, text):
        self.original_stdout.write(text); self.original_stdout.flush(); self.log_buffer.write(text)
    def flush(self): self.original_stdout.flush()
    def get_log(self): return self.log_buffer.getvalue()
    def save_log(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Experiment - {EXPERIMENT_NAME} - Optimization Log\n"); f.write(f"Started at: {time.ctime(self.start_time)}\n")
            f.write(f"Completed at: {time.ctime()}\n"); f.write(f"Total runtime: {(time.time() - self.start_time)/60:.2f} minutes\n")
            f.write("="*80 + "\n\n"); f.write(self.get_log().encode('ascii', 'ignore').decode('ascii'))

output_logger = OutputLogger()
output_logger.start_logging()

print("CNN ARCHITECTURE HYPERPARAMETER OPTIMIZATION (EXTENSIVE)")
print("="*70)
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"METHOD: Using {N_SPLITS_K_FOLD}-Fold Cross-Validation within each Optuna trial for robust evaluation.")
print(f"SEED: {SEED}")
print("="*70)

# --- Set up organized output directories ---
try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

experiment_dir = project_root / 'results' / EXPERIMENT_NAME
trial_plots_dir = experiment_dir / "trial_plots"
study_plots_dir = experiment_dir / "study_plots"
final_model_plots_dir = experiment_dir / "final_model_plots"
for d in [experiment_dir, trial_plots_dir, study_plots_dir, final_model_plots_dir]:
    d.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved in: {experiment_dir}")

data_path = project_root / 'data' / 'processed' / 'ProSeq_binary_classification.csv'
print(f"Loading data from: {data_path}")

data_binary = pd.read_csv(data_path)
data_filtered = data_binary[data_binary['ProSeq'].str.len() >= 600].copy()
print(f"Dataset size: {len(data_filtered)}")

class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(data_filtered['binary_classification']),
                                   y=data_filtered['binary_classification'])

class BinaryClassificationDataset(Dataset):
    def __init__(self, data, target_length=600):
        self.data = data.reset_index(drop=True)
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3}
        self.target_length = target_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sequence, target = self.data.iloc[idx]['ProSeq'], self.data.iloc[idx]['binary_classification']
        return self.one_hot_encode(sequence), target
    def one_hot_encode(self, sequence):
        sequence = sequence[:self.target_length]
        one_hot = np.zeros((self.target_length, 4), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.dna_dict:
                one_hot[i, self.dna_dict[nucleotide]] = 1.0
        return one_hot

# Create a single train+val split, and a final test set. K-fold will be done on train_val_data.
train_val_data, test_data = train_test_split(data_filtered, test_size=0.2, random_state=SEED, stratify=data_filtered['binary_classification'])
train_val_dataset = BinaryClassificationDataset(train_val_data)
test_dataset = BinaryClassificationDataset(test_data)
print(f"\nData splits: Train+Val={len(train_val_dataset)}, Test={len(test_dataset)}")

class FlexibleCNN(nn.Module):
    """Flexible CNN architecture for hyperparameter optimization."""
    def __init__(self, num_conv_layers, conv_channels, kernel_sizes, pool_sizes,
                 num_fc_layers, fc_sizes, dropout_rate, use_batch_norm, activation, pooling_type, input_channels=4):
        super(FlexibleCNN, self).__init__()
        self.num_conv_layers, self.num_fc_layers = num_conv_layers, num_fc_layers
        self.use_batch_norm, self.dropout_rate = use_batch_norm, dropout_rate
        self.activation = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'swish': nn.SiLU, 'elu': nn.ELU}[activation]()
        
        self.conv_layers, self.bn_layers, self.pool_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        in_channels = input_channels
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            in_channels = out_channels
        
        self.pooling_type = pooling_type
        if pooling_type == 'avg': self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max': self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_avg_pool, self.global_max_pool = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
            in_channels *= 2
        
        self.fc_layers, self.fc_bn_layers = nn.ModuleList(), nn.ModuleList()
        if num_fc_layers > 0:
            fc_input_size = in_channels
            for i in range(num_fc_layers):
                fc_output_size = fc_sizes[i]
                self.fc_layers.append(nn.Linear(fc_input_size, fc_output_size))
                self.fc_bn_layers.append(nn.BatchNorm1d(fc_output_size) if use_batch_norm else nn.Identity())
                fc_input_size = fc_output_size
            self.output_layer = nn.Linear(fc_input_size, 1)
        else:
            self.output_layer = nn.Linear(in_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.num_conv_layers):
            x = self.activation(self.bn_layers[i](self.conv_layers[i](x)))
            x = self.pool_layers[i](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling_type == 'both':
            x = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], dim=1).squeeze(-1)
        else:
            x = self.global_pool(x).squeeze(-1)
        
        for i in range(self.num_fc_layers):
            x = self.activation(self.fc_bn_layers[i](self.fc_layers[i](x)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return torch.sigmoid(self.output_layer(x)).squeeze()

def _plot_kfold_trial_curves(trial_number, fold_histories, save_path):
    """
    Plots the mean and standard deviation of training/validation metrics across K-Folds for a single trial.
    """
    if not fold_histories:
        return

    # Process histories into DataFrames, handling varying lengths due to early stopping
    metrics = ['train_loss', 'val_auc', 'val_acc']
    metric_dfs = {metric: pd.DataFrame([pd.Series(fold[metric], index=fold['epochs']) for fold in fold_histories]).T for metric in metrics}
    
    # Calculate mean and std, which handles NaNs correctly
    stats = {metric: {'mean': df.mean(axis=1), 'std': df.std(axis=1)} for metric, df in metric_dfs.items()}
    epochs = stats['train_loss']['mean'].index

    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'Trial {trial_number} K-Fold Training Curves (Mean ± Std Dev over {len(fold_histories)} Folds)', fontsize=16)

    # Plot Training Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Avg. Training Loss', color=color)
    ax1.plot(epochs, stats['train_loss']['mean'], color=color, marker='o', markersize=4, label='Mean Train Loss')
    ax1.fill_between(epochs, stats['train_loss']['mean'] - stats['train_loss']['std'], stats['train_loss']['mean'] + stats['train_loss']['std'], color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Validation Metrics
    ax2 = ax1.twinx()
    color_auc = 'tab:blue'
    ax2.set_ylabel('Validation Metric', color=color_auc)
    ax2.plot(epochs, stats['val_auc']['mean'], color=color_auc, marker='s', markersize=4, label='Mean Val AUC')
    ax2.fill_between(epochs, stats['val_auc']['mean'] - stats['val_auc']['std'], stats['val_auc']['mean'] + stats['val_auc']['std'], color=color_auc, alpha=0.2)
    
    color_acc = 'tab:green'
    ax2.plot(epochs, stats['val_acc']['mean'], color=color_acc, marker='^', markersize=4, linestyle='--', label='Mean Val Accuracy')
    ax2.fill_between(epochs, stats['val_acc']['mean'] - stats['val_acc']['std'], stats['val_acc']['mean'] + stats['val_acc']['std'], color=color_acc, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color_auc)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"    Saved K-Fold training curve plot to {save_path}")

class ArchitectureOptimizer:
    """Class for optimizing the architecture of the CNN using K-Fold Cross-Validation."""
    def __init__(self, device, full_train_dataset, class_weights):
        self.device = device
        self.full_train_dataset = full_train_dataset
        self.class_weights = class_weights
        self.labels = full_train_dataset.data['binary_classification'].values

    def _create_model_from_params(self, params: dict) -> FlexibleCNN:
        num_conv_layers, use_fc_layers = params['num_conv_layers'], params['use_fc_layers']
        num_fc_layers = params.get('num_fc_layers', 0) if use_fc_layers else 0
        conv_channels, kernel_sizes, pool_sizes = [], [], []
        base_channels = params['conv_base_channels']
        for i in range(num_conv_layers):
            multiplier = params[f'conv_multiplier_{i}']
            conv_channels.append(max(8, min(int(base_channels * (multiplier**i)), 512)))
            kernel_sizes.append(params[f'kernel_size_{i}'])
            pool_sizes.append(params[f'pool_size_{i}'])
        fc_sizes = [params[f'fc_size_{i}'] for i in range(num_fc_layers)] if use_fc_layers and num_fc_layers > 0 else []
        return FlexibleCNN(num_conv_layers=num_conv_layers, conv_channels=conv_channels, kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, num_fc_layers=num_fc_layers, fc_sizes=fc_sizes, dropout_rate=params['dropout_rate'], use_batch_norm=params['use_batch_norm'], activation=params['activation'], pooling_type=params['pooling_type'])
    
    def _create_optimizer_from_params(self, params: dict, model: nn.Module) -> optim.Optimizer:
        optimizer_type, lr, wd = params['optimizer'], params['learning_rate'], params['weight_decay']
        if optimizer_type == 'adamw':
            betas = (params.get('adam_beta1', 0.9), params.get('adam_beta2', 0.999))
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        else: # sgd
            return optim.SGD(model.parameters(), lr=lr, momentum=params.get('momentum', 0.9), weight_decay=wd)

    def objective(self, trial: optuna.Trial) -> float:
        print(f"\nTRIAL {trial.number} starting...")
        # --- Extensive Hyperparameter Space ---
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 4)
        use_fc_layers = trial.suggest_categorical('use_fc_layers', [True, False])
        num_fc_layers = trial.suggest_int('num_fc_layers', 1, 2) if use_fc_layers else 0
        trial.set_user_attr('num_fc_layers_actual', num_fc_layers)
        trial.suggest_int('conv_base_channels', 8, 128)
        for i in range(num_conv_layers):
            trial.suggest_float(f'conv_multiplier_{i}', 0.75, 2.5)
            trial.suggest_int(f'kernel_size_{i}', 3, 15, step=2)
            trial.suggest_int(f'pool_size_{i}', 2, 4)
        if use_fc_layers:
            for i in range(num_fc_layers):
                trial.suggest_int(f'fc_size_{i}', 32, 512)
        trial.suggest_float('dropout_rate', 0.1, 0.8)
        trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        trial.suggest_categorical('use_batch_norm', [True, False])
        trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu'])
        optimizer_type = trial.suggest_categorical('optimizer', ['adamw', 'sgd'])
        if optimizer_type == 'adamw':
            trial.suggest_float('adam_beta1', 0.85, 0.95)
            trial.suggest_float('adam_beta2', 0.99, 0.9999)
        if optimizer_type == 'sgd':
            trial.suggest_float('momentum', 0.8, 0.99)
        trial.suggest_categorical('pooling_type', ['avg', 'max', 'both'])

        kf = StratifiedKFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=SEED)
        fold_scores, fold_histories = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(self.full_train_dataset)), self.labels)):
            print(f"  --- Fold {fold+1}/{N_SPLITS_K_FOLD} ---")
            
            try:
                model = self._create_model_from_params(trial.params).to(self.device)
                optimizer = self._create_optimizer_from_params(trial.params, model)
                train_subset, val_subset = Subset(self.full_train_dataset, train_idx), Subset(self.full_train_dataset, val_idx)
                fold_train_labels = self.labels[train_idx]
                sample_weights = np.array([self.class_weights[t] for t in fold_train_labels])
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                train_loader = DataLoader(train_subset, batch_size=trial.params['batch_size'], sampler=sampler)
                val_loader = DataLoader(val_subset, batch_size=trial.params['batch_size'], shuffle=False)
                criterion = nn.BCELoss()
                max_epochs, patience = 80, 10
                best_val_auc, patience_counter = 0.0, 0
                history = {'epochs': [], 'train_loss': [], 'val_auc': [], 'val_acc': []}

                for epoch in range(max_epochs):
                    model.train(); epoch_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)
                        optimizer.zero_grad(); outputs = model(batch_x)
                        with torch.no_grad():
                            true_dist = batch_y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                        loss = criterion(outputs, true_dist)
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
                        epoch_loss += loss.item()
                    
                    # Log metrics at a fixed interval
                    if (epoch + 1) % 3 == 0:
                        avg_train_loss = epoch_loss / len(train_loader)
                        model.eval(); val_probs_es, val_targets_es = [], []
                        with torch.no_grad():
                            for batch_x, batch_y in val_loader:
                                outputs = model(batch_x.to(self.device))
                                val_probs_es.extend(np.atleast_1d(outputs.cpu().numpy()))
                                val_targets_es.extend(np.atleast_1d(batch_y.cpu().numpy()))
                        
                        val_preds_es = (np.array(val_probs_es) > 0.5).astype(int)
                        val_auc_es = roc_auc_score(val_targets_es, val_probs_es)
                        val_acc_es = accuracy_score(val_targets_es, val_preds_es)

                        history['epochs'].append(epoch + 1); history['train_loss'].append(avg_train_loss)
                        history['val_auc'].append(val_auc_es); history['val_acc'].append(val_acc_es)
                        
                        if val_auc_es > best_val_auc:
                            best_val_auc = val_auc_es
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        if patience_counter >= patience:
                            print(f"    Early stopping at epoch {epoch+1} with best Val AUC: {best_val_auc:.4f}")
                            break
                
                fold_scores.append(best_val_auc)
                fold_histories.append(history)
                trial.report(np.mean(fold_scores), fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            except Exception as e:
                print(f"  ERROR in fold {fold+1}: {e}. Skipping trial.")
                return 0.0

        _plot_kfold_trial_curves(trial.number, fold_histories, trial_plots_dir / f"trial_{trial.number}_curves.png")

        avg_auc = np.mean(fold_scores)
        trial.set_user_attr('param_count', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(f"  TRIAL {trial.number} COMPLETED. Avg K-Fold AUC: {avg_auc:.4f}")
        return avg_auc
    
    def optimize(self, n_trials=100, timeout=7200):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        return study
    
    def visualize_optimization(self, study, save_dir):
        print("\nCreating optimization summary visualizations...")
        try:
            plots = {'history': optuna.visualization.plot_optimization_history, 'param_importances': optuna.visualization.plot_param_importances, 'slice': optuna.visualization.plot_slice, 'contour': optuna.visualization.plot_contour}
            for name, plot_func in plots.items():
                fig = plot_func(study)
                save_path = save_dir / f"study_{name}.png"
                fig.write_image(save_path, scale=2)
                print(f"  Saved {name} plot to {save_path}")
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"\n[WARNING] Could not generate some Optuna plots (this is common if a parameter has low importance). Please `pip install plotly kaleido`.")

def save_study_results_to_csv(study, save_path):
    print(f"\nSaving full study results to CSV...")
    df = study.trials_dataframe()
    df.to_csv(save_path, index=False)
    print(f"  Successfully saved results for {len(df)} trials to {save_path}")

def plot_final_evaluation_results(results, save_dir):
    print("\nCreating final model evaluation plots...")
    cm_path = save_dir / get_timestamped_filename('final_model_confusion_matrix', 'png')
    plt.figure(figsize=(8, 6)); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f"Final Model Confusion Matrix\nAccuracy: {results['test_accuracy']:.4f}", fontsize=14); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(cm_path); plt.close(); print(f"  Saved confusion matrix plot to {cm_path}")

    roc_path = save_dir / get_timestamped_filename('final_model_roc_curve', 'png')
    fpr, tpr, _ = roc_curve(results['test_targets'], results['test_probs'])
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {results['test_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Final Model Receiver Operating Characteristic (ROC) Curve', fontsize=14); plt.legend(loc="lower right"); plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(roc_path); plt.close(); print(f"  Saved ROC curve plot to {roc_path}")
    return [cm_path, roc_path]

def evaluate_best_model(optimizer, study, full_train_dataset, test_dataset, device):
    print(f"\nEVALUATING BEST MODEL ON TEST SET")
    best_params = study.best_params
    model = optimizer._create_model_from_params(best_params).to(device)
    optim = optimizer._create_optimizer_from_params(best_params, model)
    
    train_targets = full_train_dataset.data['binary_classification'].values
    sample_weights = np.array([class_weights[t] for t in train_targets])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(full_train_dataset, batch_size=best_params['batch_size'], sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    criterion = nn.BCELoss()
    num_epochs = 80
    print(f"Training best model for {num_epochs} epochs on the full training dataset...")
    
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
            optim.zero_grad(); outputs = model(batch_x); loss = criterion(outputs, batch_y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optim.step()
        if (epoch + 1) % 10 == 0: print(f"  Final Training Epoch {epoch+1}/{num_epochs}")
    
    model.eval(); test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device))
            test_probs.extend(np.atleast_1d(outputs.cpu().numpy()))
            test_targets.extend(np.atleast_1d(batch_y.cpu().numpy()))
    test_preds = (np.array(test_probs) > 0.5).astype(int)

    results = {'best_params': best_params, 'test_accuracy': accuracy_score(test_targets, test_preds), 'test_f1': f1_score(test_targets, test_preds),
               'test_auc': roc_auc_score(test_targets, test_probs), 'confusion_matrix': confusion_matrix(test_targets, test_preds).tolist(),
               'test_targets': test_targets, 'test_probs': test_probs, 'param_count': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    print("\nFINAL TEST RESULTS:"); print(f"  Test Accuracy: {results['test_accuracy']:.4f}, F1: {results['test_f1']:.4f}, AUC: {results['test_auc']:.4f}")
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    study, final_results = None, None
    try:
        optimizer = ArchitectureOptimizer(device, train_val_dataset, class_weights)
        study = optimizer.optimize(n_trials=100, timeout=7200)
        
        optimizer.visualize_optimization(study, study_plots_dir)
        
        csv_path = experiment_dir / get_timestamped_filename(f'{EXPERIMENT_NAME}_study_results', 'csv')
        save_study_results_to_csv(study, csv_path)

        final_results = evaluate_best_model(optimizer, study, train_val_dataset, test_dataset, device)
        final_plots_paths = plot_final_evaluation_results(final_results, final_model_plots_dir)
        
        del final_results['test_targets']; del final_results['test_probs']
        results_to_save = {'study_summary': {'best_value': study.best_value, 'best_params': study.best_params}, 'final_evaluation': final_results}
        json_path = experiment_dir / get_timestamped_filename(f'{EXPERIMENT_NAME}_results_summary', 'json')
        with open(json_path, 'w') as f: json.dump(results_to_save, f, indent=2, default=str)
        
    except Exception as e:
        print(f"\nERROR OCCURRED: {e}")
        import traceback; traceback.print_exc()
        
    finally:
        log_path = experiment_dir / get_timestamped_filename(f'{EXPERIMENT_NAME}_log', 'txt')
        output_logger.save_log(log_path)
        print(f"Complete training log saved to '{log_path}'")
        output_logger.stop_logging()
        
        print("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)
        if study and study.best_trial:
            print(f"Trials completed: {len(study.trials)}")
            print(f"Best validation K-Fold AUC: {study.best_value:.4f}")
            if final_results:
                print(f"Final Test Accuracy: {final_results['test_accuracy']:.4f}, Test F1: {final_results['test_f1']:.4f}")
            print(f"Best Architecture: {study.best_params['num_conv_layers']} conv, {study.best_trial.user_attrs.get('num_fc_layers_actual', 0)} FC")
        
        print(f"\nFILES CREATED IN: {experiment_dir}")
        print(f"  - Log File: {log_path.name}")
        if 'json_path' in locals(): print(f"  - Results JSON: {json_path.name}")
        if 'csv_path' in locals(): print(f"  - Study CSV: {csv_path.name}")
        print(f"  - Study Plots saved in: {study_plots_dir.name}/")
        print(f"  - Final Model Plots saved in: {final_model_plots_dir.name}/")
        print(f"  - Per-Trial Plots saved in: {trial_plots_dir.name}/")
        print("="*80)