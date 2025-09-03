"""
CNN Architecture Hyperparameter Optimization for Binary Classification
Optimizes number of convolutional layers, kernel sizes, and fully connected layers
Uses Optuna for Bayesian optimization with effective pruning and generates extensive visualizations.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import json
import time
import random
from typing import Dict, List, Tuple, Optional
import warnings
import sys
from io import StringIO
from datetime import datetime
from pathlib import Path

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

def get_timestamped_filename(base_name, extension):
    """Generate a timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

class OutputLogger:
    """Captures all print outputs for logging"""
    def __init__(self):
        self.log_buffer = StringIO()
        self.original_stdout = sys.stdout
        self.start_time = time.time()
        
    def start_logging(self): sys.stdout = self
    def stop_logging(self): sys.stdout = self.original_stdout
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        self.log_buffer.write(text)
    def flush(self): self.original_stdout.flush()
    def get_log(self): return self.log_buffer.getvalue()
    def save_log(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"CNN Architecture Optimization Log\n")
            f.write(f"Started at: {time.ctime(self.start_time)}\n")
            f.write(f"Completed at: {time.ctime()}\n")
            f.write(f"Total runtime: {(time.time() - self.start_time)/60:.2f} minutes\n")
            f.write("="*80 + "\n\n")
            log_content = self.get_log().encode('ascii', 'ignore').decode('ascii')
            f.write(log_content)

output_logger = OutputLogger()
output_logger.start_logging()

print("CNN ARCHITECTURE HYPERPARAMETER OPTIMIZATION")
print("="*70)
print("TARGET: Optimizing Conv layers, kernel sizes, FC layers")
print("METHOD: Using Optuna Bayesian optimization with pruning")
print("PLOTTING: Generating per-trial curves, study summary plots, and final model evaluation graphs.")
print(f"OUTPUT: Files will be timestamped: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
print("-" * 70)

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    print("WARNING: __file__ not defined. Assuming current directory is project root.")
    project_root = Path.cwd()

results_dir = project_root / 'results'
trial_plots_dir = results_dir / "trial_plots" # Directory for individual trial plots
results_dir.mkdir(exist_ok=True)
trial_plots_dir.mkdir(exist_ok=True)

data_path = project_root / 'data' / 'processed' / 'ProSeq_binary_classification.csv'
print(f"Loading data from: {data_path}")

data_binary = pd.read_csv(data_path)
data_binary = data_binary[['binary_classification', 'ProSeq']]

data_filtered = data_binary[data_binary['ProSeq'].str.len() >= 600].copy()
print(f"Dataset size: {len(data_filtered)}")

class_dist = data_filtered['binary_classification'].value_counts()
print(f"\nClass distribution:")
for class_val, count in class_dist.items():
    print(f"  Class {class_val}: {count} ({count / len(data_filtered) * 100:.1f}%)")

class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(data_filtered['binary_classification']),
                                   y=data_filtered['binary_classification'])
print(f"Balanced class weights: {class_weights}")

class BinaryClassificationDataset(Dataset):
    def __init__(self, data, target_length=600):
        self.data = data.reset_index(drop=True)
        self.dna_dict = {"A": 0, "T": 1, "G": 2, "C": 3}
        self.target_length = target_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['ProSeq']
        target = self.data.iloc[idx]['binary_classification']
        return self.one_hot_encode(sequence), target
    def one_hot_encode(self, sequence):
        sequence = sequence[:self.target_length]
        one_hot = np.zeros((self.target_length, 4), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.dna_dict:
                one_hot[i, self.dna_dict[nucleotide]] = 1.0
        return one_hot

train_data, test_data = train_test_split(data_filtered, test_size=0.2, random_state=42, stratify=data_filtered['binary_classification'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['binary_classification'])
print(f"\nData splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

train_dataset = BinaryClassificationDataset(train_data)
val_dataset = BinaryClassificationDataset(val_data)
test_dataset = BinaryClassificationDataset(test_data)

class FlexibleCNN(nn.Module):
    """Flexible CNN architecture with variable number of layers"""
    def __init__(self, num_conv_layers, conv_channels, kernel_sizes, pool_sizes,
                 num_fc_layers, fc_sizes, dropout_rate, use_batch_norm, activation, pooling_type, input_channels=4):
        super(FlexibleCNN, self).__init__()
        self.num_conv_layers, self.num_fc_layers = num_conv_layers, num_fc_layers
        self.use_batch_norm, self.dropout_rate = use_batch_norm, dropout_rate
        self.activation = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'swish': nn.SiLU, 'elu': nn.ELU}[activation]()
        
        # Conv layers
        self.conv_layers, self.bn_layers, self.pool_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        in_channels = input_channels
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            in_channels = out_channels
        
        # Global pooling
        self.pooling_type = pooling_type
        if pooling_type == 'avg': self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max': self.global_pool = nn.AdaptiveMaxPool1d(1)
        else: # 'both'
            self.global_avg_pool, self.global_max_pool = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
            in_channels *= 2
        
        # FC layers
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
        
        if self.pooling_type == 'both': x = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], dim=1).squeeze(-1)
        else: x = self.global_pool(x).squeeze(-1)
        
        for i in range(self.num_fc_layers):
            x = self.activation(self.fc_bn_layers[i](self.fc_layers[i](x)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return torch.sigmoid(self.output_layer(x)).squeeze()

# --- [NEW] Plotting function for individual trial curves ---
def _plot_trial_curves(trial_number, history, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Trial {trial_number} Training Curves', fontsize=16)

    # Plot Training Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Avg. Training Loss', color=color)
    ax1.plot(history['epochs'], history['train_loss'], color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Validation Metrics on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Metric', color=color)
    ax2.plot(history['epochs'], history['val_auc'], color=color, marker='s', linestyle='--', label='Val AUC')
    ax2.plot(history['epochs'], history['val_acc'], color='tab:green', marker='^', linestyle=':', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig) # Prevent memory leaks
    print(f"    Saved training curve plot to {save_path}")

class ArchitectureOptimizer:
    def __init__(self, device, train_dataset, val_dataset, class_weights):
        self.device = device
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.class_weights = class_weights
        self.train_targets = train_data['binary_classification'].values

    def _create_model_from_params(self, params):
        # ... (implementation from previous version, unchanged) ...
        num_conv_layers = params['num_conv_layers']
        use_fc_layers = params['use_fc_layers']
        num_fc_layers = params.get('num_fc_layers', 0) if use_fc_layers else 0

        conv_channels, kernel_sizes, pool_sizes = [], [], []
        base_channels = params['conv_base_channels']
        for i in range(num_conv_layers):
            multiplier = params[f'conv_multiplier_{i}']
            channels = int(base_channels * (multiplier ** i))
            conv_channels.append(max(8, min(channels, 1024)))
            kernel_sizes.append(params[f'kernel_size_{i}'])
            pool_sizes.append(params[f'pool_size_{i}'])
        
        fc_sizes = [params[f'fc_size_{i}'] for i in range(num_fc_layers)] if use_fc_layers and num_fc_layers > 0 else []
        
        return FlexibleCNN(
            num_conv_layers=num_conv_layers, conv_channels=conv_channels, kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes, num_fc_layers=num_fc_layers, fc_sizes=fc_sizes,
            dropout_rate=params['dropout_rate'], use_batch_norm=params['use_batch_norm'],
            activation=params['activation'], pooling_type=params['pooling_type']
        )
    
    def _create_optimizer_from_params(self, params, model):
        # ... (implementation from previous version, unchanged) ...
        optimizer_type, lr, wd = params['optimizer'], params['learning_rate'], params['weight_decay']
        if optimizer_type == 'adam': return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == 'adamw': return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == 'rmsprop': return optim.RMSprop(model.parameters(), lr=lr, alpha=params.get('rmsprop_alpha', 0.99), weight_decay=wd)
        else: return optim.SGD(model.parameters(), lr=lr, momentum=params.get('momentum', 0.9), weight_decay=wd)

    def create_balanced_loader(self, dataset, targets, batch_size):
        sample_weights = np.array([self.class_weights[t] for t in targets])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def objective(self, trial):
        print(f"\nTRIAL {trial.number} starting...")
        # ... (hyperparameter suggestion logic is unchanged) ...
        num_conv_layers = trial.suggest_int('num_conv_layers', 2, 5)
        use_fc_layers = trial.suggest_categorical('use_fc_layers', [True, False])
        num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3) if use_fc_layers else 0
        trial.set_user_attr('num_fc_layers_actual', num_fc_layers)
        
        base_channels = trial.suggest_int('conv_base_channels', 8, 256)
        for i in range(num_conv_layers):
            trial.suggest_float(f'conv_multiplier_{i}', 0.5, 4.0)
            trial.suggest_int(f'kernel_size_{i}', 3, 21, step=2)
            trial.suggest_int(f'pool_size_{i}', 2, 6)
        
        if use_fc_layers:
            for i in range(num_fc_layers):
                min_size, max_size = max(16, 1024 // (2 ** (i + 1))), min(2048, 1024 // (2 ** i))
                trial.suggest_int(f'fc_size_{i}', min_size, max_size)
        
        trial.suggest_float('dropout_rate', 0.0, 0.8)
        trial.suggest_float('learning_rate', 1e-6, 5e-2, log=True)
        trial.suggest_float('weight_decay', 1e-8, 1e-1, log=True)
        trial.suggest_int('batch_size', 8, 256, step=8)
        trial.suggest_categorical('use_batch_norm', [True, False])
        trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'swish', 'elu'])
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd', 'rmsprop'])
        
        if optimizer_type == 'rmsprop': trial.suggest_float('rmsprop_alpha', 0.9, 0.999)
        if optimizer_type == 'sgd': trial.suggest_float('momentum', 0.5, 0.99)
        trial.suggest_categorical('pooling_type', ['avg', 'max', 'both'])
        
        try:
            model = self._create_model_from_params(trial.params).to(self.device)
            optimizer = self._create_optimizer_from_params(trial.params, model)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Architecture: {num_conv_layers} conv, {num_fc_layers} FC. Params: {param_count:,}")
            if param_count > 5_000_000: return 0.0

            batch_size = trial.params['batch_size']
            train_loader = self.create_balanced_loader(self.train_dataset, self.train_targets, batch_size)
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
            
            criterion = nn.BCELoss()
            max_epochs, patience = 100, 15
            best_val_auc, patience_counter, best_model_weights, best_epoch = 0.0, 0, None, 0
            
            # --- [MODIFIED] History tracking for plotting ---
            history = {'epochs': [], 'train_loss': [], 'val_auc': [], 'val_acc': []}
            
            for epoch in range(max_epochs):
                model.train()
                epoch_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                
                if (epoch + 1) % 3 == 0:
                    model.eval()
                    val_probs_es, val_targets_es = [], []
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            outputs = model(batch_x.to(self.device))
                            val_probs_es.extend(outputs.cpu().numpy())
                            val_targets_es.extend(batch_y.cpu().numpy())
                    
                    val_preds_es = (np.array(val_probs_es) > 0.5).astype(int)
                    val_auc_es = roc_auc_score(val_targets_es, val_probs_es)
                    val_acc_es = accuracy_score(val_targets_es, val_preds_es)
                    
                    # --- [MODIFIED] Append to history ---
                    history['epochs'].append(epoch + 1)
                    history['train_loss'].append(avg_train_loss)
                    history['val_auc'].append(val_auc_es)
                    history['val_acc'].append(val_acc_es)

                    print(f"    Epoch {epoch+1}/{max_epochs}: Train Loss={avg_train_loss:.4f}, Val AUC={val_auc_es:.4f}, Val Acc={val_acc_es:.4f}")
                    
                    trial.report(val_auc_es, epoch)
                    if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                    
                    if val_auc_es > best_val_auc:
                        best_val_auc, patience_counter, best_epoch = val_auc_es, 0, epoch + 1
                        best_model_weights = model.state_dict().copy()
                    else: patience_counter += 1
                    
                    if patience_counter >= patience: print(f"    Early stopping at epoch {epoch+1}"); break
            
            # --- [MODIFIED] Plot trial curves after successful completion ---
            if best_model_weights:
                _plot_trial_curves(trial.number, history, trial_plots_dir / f"trial_{trial.number}_curves.png")
                model.load_state_dict(best_model_weights)
            else: # Handle cases where training finishes without improving
                print("    Warning: No best model found during training.")
                return 0.0

            val_f1 = f1_score(val_targets_es, val_preds_es) if len(set(val_preds_es)) > 1 else 0.0
            print(f"  Results: F1={val_f1:.4f}, Best AUC={best_val_auc:.4f}, Best Acc={val_acc_es:.4f}")
            
            trial.set_user_attr('val_accuracy', val_acc_es); trial.set_user_attr('val_f1', val_f1)
            trial.set_user_attr('val_auc', best_val_auc); trial.set_user_attr('param_count', param_count)
            trial.set_user_attr('best_epoch', best_epoch)
            
            print(f"  TRIAL {trial.number} COMPLETED. Objective (AUC): {best_val_auc:.4f}")
            return best_val_auc
            
        except optuna.exceptions.TrialPruned: print("    Trial pruned."); raise
        except Exception as e: print(f"  ERROR: Trial {trial.number} failed: {e}"); return 0.0
    
    def optimize(self, n_trials=100, timeout=3600):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, n_min_trials=5), sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        return study
    
    # --- [MODIFIED] Switched to Optuna's built-in visualization ---
    def visualize_optimization(self, study, base_save_path):
        print("\nCreating optimization summary visualizations...")
        try:
            plots = {
                'history': optuna.visualization.plot_optimization_history,
                'param_importances': optuna.visualization.plot_param_importances,
                'slice': optuna.visualization.plot_slice,
                'contour': optuna.visualization.plot_contour,
            }
            for name, plot_func in plots.items():
                fig = plot_func(study)
                save_path = base_save_path.with_name(f"{base_save_path.stem}_{name}.png")
                fig.write_image(save_path, scale=2) # Higher resolution
                print(f"  Saved {name} plot to {save_path}")
        except (ImportError, RuntimeError) as e:
            print(f"\n[WARNING] Could not generate Optuna plots. Please `pip install plotly kaleido` to enable them.")
            print(f"Error details: {e}\n")

# --- [NEW] Plotting for final model evaluation ---
def plot_final_evaluation_results(results, save_dir):
    print("\nCreating final model evaluation plots...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f"Final Model Confusion Matrix\nAccuracy: {results['test_accuracy']:.4f}", fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = save_dir / get_timestamped_filename('final_model_confusion_matrix', 'png')
    plt.savefig(cm_path)
    plt.close()
    print(f"  Saved confusion matrix plot to {cm_path}")

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(results['test_targets'], results['test_probs'])
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {results['test_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final Model Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    roc_path = save_dir / get_timestamped_filename('final_model_roc_curve', 'png')
    plt.savefig(roc_path)
    plt.close()
    print(f"  Saved ROC curve plot to {roc_path}")
    return [cm_path, roc_path]

def evaluate_best_model(optimizer, study, train_dataset, val_dataset, test_dataset, device):
    print(f"\nEVALUATING BEST MODEL ON TEST SET")
    best_params = study.best_params
    model = optimizer._create_model_from_params(best_params).to(device)
    optim = optimizer._create_optimizer_from_params(best_params, model)
    
    print("Combining train and validation sets for final training...")
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    combined_targets = np.concatenate([train_data['binary_classification'].values, val_data['binary_classification'].values])
    
    train_loader = optimizer.create_balanced_loader(combined_dataset, combined_targets, best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    criterion = nn.BCELoss()
    num_epochs = study.best_trial.user_attrs.get('best_epoch', 50)
    print(f"Training best model for a fixed {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
            optim.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        if (epoch + 1) % 5 == 0: print(f"  Epoch {epoch+1}/{num_epochs}")
    
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device))
            test_probs.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    test_preds = (np.array(test_probs) > 0.5).astype(int)

    results = {
        'best_params': best_params,
        'test_accuracy': accuracy_score(test_targets, test_preds),
        'test_f1': f1_score(test_targets, test_preds),
        'test_auc': roc_auc_score(test_targets, test_probs),
        'confusion_matrix': confusion_matrix(test_targets, test_preds).tolist(),
        'test_targets': test_targets, 'test_probs': test_probs, # For plotting
        'param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print("\nFINAL TEST RESULTS:")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Test F1 Score: {results['test_f1']:.4f}")
    print(f"  Test AUC Score: {results['test_auc']:.4f}")
    print(f"  Model parameters: {results['param_count']:,}")
    return results

# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    study, final_results = None, None
    error_occurred, error_message = False, ""
    log_path, results_path, viz_path, final_plots_paths = None, None, None, []
    
    try:
        optimizer = ArchitectureOptimizer(device, train_dataset, val_dataset, class_weights)
        study = optimizer.optimize(n_trials=100, timeout=3600)
        
        viz_path = results_dir / get_timestamped_filename('cnn_optimization_summary', 'png')
        optimizer.visualize_optimization(study, viz_path)
        
        final_results = evaluate_best_model(optimizer, study, train_dataset, val_dataset, test_dataset, device)
        final_plots_paths = plot_final_evaluation_results(final_results, results_dir)
        
        # Clean up results dict for JSON saving
        del final_results['test_targets']
        del final_results['test_probs']

        results_to_save = {'study_summary': {'best_value': study.best_value, 'best_params': study.best_params}, 'final_evaluation': final_results}
        results_path = results_dir / get_timestamped_filename('cnn_optimization_results', 'json')
        with open(results_path, 'w') as f: json.dump(results_to_save, f, indent=2, default=str)
        print(f"\nResults summary saved to '{results_path}'")
        
    except Exception as e:
        error_occurred, error_message = True, str(e)
        print(f"\nERROR OCCURRED: {error_message}")
        import traceback; traceback.print_exc()
        
    finally:
        log_path = results_dir / get_timestamped_filename('cnn_optimization_log', 'txt')
        output_logger.save_log(log_path)
        print(f"Complete training log saved to '{log_path}'")
        output_logger.stop_logging()
        
        print("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)
        print(f"Total runtime: {(time.time() - output_logger.start_time)/60:.2f} minutes")
        if error_occurred: print(f"Error occurred: {error_message}")
        
        if study and study.best_trial:
            print(f"Trials completed: {len(study.trials)}")
            print(f"Best validation AUC: {study.best_value:.4f}")
            if final_results:
                print(f"Final Test Accuracy: {final_results['test_accuracy']:.4f}, Test F1: {final_results['test_f1']:.4f}")
            print(f"Best Architecture: {study.best_params['num_conv_layers']} conv, {study.best_trial.user_attrs.get('num_fc_layers_actual', 0)} FC")
        
        print(f"\nFILES CREATED:")
        if results_path: print(f"  Results JSON: {results_path}")
        if log_path: print(f"  Log File: {log_path}")
        if viz_path: print(f"  Study Plots: {viz_path.parent / (viz_path.stem + '_*')}")
        if final_plots_paths:
            for p in final_plots_paths: print(f"  Final Model Plot: {p}")
        print(f"  Per-Trial Plots: {trial_plots_dir}/")
        print("="*80)