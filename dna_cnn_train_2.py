#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA CNN with Hyperparameter Tuning and Advanced Analysis (v2 - Patched)

- Fix for DataLoader collation error (numpy.int64 issue).
- Hyperparameter tuning using Optuna for lr, weight decay, dropout, loss, and batch size.
- Advanced CNN (DeepMotifCNN) with multi-scale kernels, residual connections, and attention.
- Professional-quality visualizations using Seaborn.
- Comprehensive final analysis report.
- Fully compatible with CUDA, MPS, and CPU.
- Saves results for each trial and a copy of the best trial's results.
"""

import argparse, os, random, math, shutil
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

import optuna

# ---------------- Device ----------------
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ---------------- IUPAC & helpers ----------------
IUPAC = {
    "A":[1,0,0,0], "C":[0,1,0,0], "G":[0,0,1,0], "T":[0,0,0,1], "N":[0.25,0.25,0.25,0.25],
    "R":[0.5,0,0.5,0], "Y":[0,0.5,0,0.5], "S":[0,0.5,0.5,0], "W":[0.5,0,0,0.5],
    "K":[0,0,0.5,0.5], "M":[0.5,0.5,0,0], "B":[0,1/3,1/3,1/3], "D":[1/3,0,1/3,1/3],
    "H":[1/3,1/3,0,1/3], "V":[1/3,1/3,1/3,0]
}
IUPAC_KEYS = set(IUPAC.keys())
RC_TABLE = str.maketrans({"A":"T","C":"G","G":"C","T":"A","N":"N","R":"Y","Y":"R","S":"S","W":"W","K":"M","M":"K","B":"V","V":"B","D":"H","H":"D"})

def clean_seq(seq: object, L: int) -> str:
    s = str(seq).upper() if seq is not None and not (isinstance(seq, float) and math.isnan(seq)) else ""
    s = "".join(c if c in IUPAC_KEYS else "N" for c in s)
    return s.ljust(L, "N")[:L]

def rev_comp_str(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]

def one_hot_mono(seq: str, L: int) -> np.ndarray:
    arr = np.zeros((4, L), dtype=np.float32)
    for i, c in enumerate(seq[:L]):
        arr[:, i] = IUPAC.get(c, IUPAC["N"])[:4] # Use first 4 elements for ACGT
    return arr

# ---------------- Dataset ----------------
class DNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=600, augment=False, rc_prob=0.5):
        self.seq_len, self.augment, self.rc_prob = seq_len, augment, rc_prob
        df["ProSeq"] = df["ProSeq"].apply(lambda x: clean_seq(x, self.seq_len))
        if "Predicted_Component_5" in df.columns:
            labels = df["Predicted_Component_5"].astype(int).values - 1
        else:
            prob_cols = [c for c in df.columns if c.endswith("_Probability")]
            labels = df[prob_cols].values.argmax(axis=1) if prob_cols else np.zeros(len(df))
        self.labels, self.seqs = labels.astype(int), df["ProSeq"].tolist()

    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s, y = self.seqs[idx], self.labels[idx]
        if self.augment and random.random() < self.rc_prob: s = rev_comp_str(s)
        x = torch.as_tensor(one_hot_mono(s, self.seq_len), dtype=torch.float32)
        # FIX: Explicitly cast the label to a standard Python int to prevent collation errors.
        return x, int(y)

# ---------------- Model: DeepMotifCNN ----------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y

class MultiScaleResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop):
        super().__init__()
        branch_ch = out_ch // 3
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1),
            nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3),
            nn.Conv1d(in_ch, branch_ch, kernel_size=11, padding=5)])
        self.merge = nn.Sequential(nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True), nn.Dropout(p_drop))
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res_path = self.shortcut(x)
        x_cat = torch.cat([conv(x) for conv in self.convs], dim=1)
        return F.relu(self.se(self.merge(x_cat)) + res_path)

class DeepMotifCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, p_drop: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 48, kernel_size=15, padding=7), nn.BatchNorm1d(48),
            nn.ReLU(inplace=True), nn.MaxPool1d(3))
        self.block1 = MultiScaleResidualSEBlock(48, 96, p_drop)
        self.pool1 = nn.MaxPool1d(4)
        self.block2 = MultiScaleResidualSEBlock(96, 120, p_drop)
        self.pool2 = nn.MaxPool1d(4)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(120, 64),
            nn.ReLU(inplace=True), nn.Dropout(p_drop + 0.1), nn.Linear(64, n_classes))
    def forward(self, x):
        return self.head(self.pool2(self.block2(self.pool1(self.block1(self.stem(x))))))

# ---------------- Losses & RC TTA ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt, loss = torch.exp(-ce), (1 - pt) ** self.gamma * ce
        return loss.mean()

def class_weighted_ce(cls_counts, device):
    w = 1.0 / torch.clamp(cls_counts.float(), min=1.0)
    w = (w / w.mean()).to(device)
    return nn.CrossEntropyLoss(weight=w)

def rc_onehot_mono(x): return torch.flip(x, dims=[1, 2])

# ---------------- Visualization ----------------
def generate_plots(history_df, cm, n_classes, out_dir):
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training & Validation Metrics', fontsize=20)
    
    best_epoch = history_df['val_loss'].idxmin()
    axes[0, 0].plot(history_df['train_loss'], label='Train Loss', lw=2)
    axes[0, 0].plot(history_df['val_loss'], label='Validation Loss', lw=2)
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
    axes[0, 0].set_title('Model Loss', fontsize=14); axes[0, 0].set_xlabel('Epochs'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend()
    
    axes[0, 1].plot(history_df['val_acc'], label='Validation Accuracy', color='green', lw=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14); axes[0, 1].set_xlabel('Epochs'); axes[0, 1].set_ylabel('Accuracy')
    
    axes[1, 0].plot(history_df['val_f1'], label='Validation Macro F1', color='purple', lw=2)
    axes[1, 0].set_title('Validation Macro F1 Score', fontsize=14); axes[1, 0].set_xlabel('Epochs'); axes[1, 0].set_ylabel('F1 Score')

    axes[1, 1].plot(1 - history_df['val_acc'], label='Validation Error Rate', color='orange', lw=2)
    axes[1, 1].set_title('Validation Error Rate', fontsize=14); axes[1, 1].set_xlabel('Epochs'); axes[1, 1].set_ylabel('Error Rate')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(out_dir, "metrics_dashboard.png")); plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(n_classes)))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title('Validation Confusion Matrix', fontsize=16); plt.savefig(os.path.join(out_dir, "confusion_matrix.png")); plt.close()

# ---------------- Training Function for One Trial ----------------
def run_training_trial(params: Dict, trial_num: int, is_best_trial: bool = False):
    out_dir = f"results/trial_{trial_num}"
    os.makedirs(out_dir, exist_ok=True)
    
    device, seed = get_device(), 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    df = pd.read_csv(params['csv'])
    labels_all = df["Predicted_Component_5"].astype(int).values - 1
    df_tr, df_va = train_test_split(df, test_size=0.2, random_state=seed, stratify=labels_all)
    
    train_ds = DNADataset(df_tr, seq_len=params['seq_len'], augment=True)
    val_ds = DNADataset(df_va, seq_len=params['seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    n_classes = int(labels_all.max() + 1)
    cls_counts = torch.tensor(np.bincount(train_ds.labels, minlength=n_classes))

    model = DeepMotifCNN(4, n_classes, p_drop=params['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'] * len(train_loader))
    criterion = FocalLoss() if params['loss'] == 'focal' else class_weighted_ce(cls_counts, device)
    
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_acc, best_metrics = 0.0, None

    for epoch in range(1, params['epochs'] + 1):
        model.train()
        train_loss_epoch = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb), criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step(); scheduler.step()
            train_loss_epoch += loss.item()
        history['train_loss'].append(train_loss_epoch / len(train_loader))

        model.eval()
        val_loss_epoch, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = 0.5 * (model(xb) + model(rc_onehot_mono(xb))) if params['rc_tta'] else model(xb)
                loss = criterion(logits, yb)
                val_loss_epoch += loss.item()
                preds = logits.argmax(1)
                correct += (preds == yb).sum().item(); total += yb.size(0)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(yb.cpu().numpy())

        acc = correct / total
        history['val_loss'].append(val_loss_epoch / len(val_loader))
        history['val_acc'].append(acc)
        history['val_f1'].append(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
            best_metrics = {
                'cm': confusion_matrix(all_labels, all_preds, labels=list(range(n_classes))),
                'report': classification_report(all_labels, all_preds, digits=4, zero_division=0, target_names=[f"Class {i}" for i in range(n_classes)])
            }
        
        if epoch % 20 == 0 or epoch == 1: print(f"Trial {trial_num}, Epoch {epoch:3d}/{params['epochs']} | Val Acc: {acc:.4f} | Best Acc: {best_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    if best_metrics:
        generate_plots(pd.DataFrame(history), best_metrics['cm'], n_classes, out_dir)
        with open(os.path.join(out_dir, "class_report.txt"), "w") as f: f.write(best_metrics['report'])

    if is_best_trial: shutil.copytree(out_dir, "results/best_trial", dirs_exist_ok=True)
        
    return best_acc

# ---------------- Optuna Objective Function ----------------
def objective(trial: optuna.trial.Trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'loss': trial.suggest_categorical('loss', ['weighted_ce', 'focal']),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'csv': "data/processed/ProSeq_with_5component_analysis.csv", 'epochs': 200, 'seq_len': 600, 'rc_tta': True
    }
    print(f"\n--- Starting Trial {trial.number} with params: { {k:v for k,v in params.items() if k not in ['csv', 'epochs', 'seq_len', 'rc_tta']} } ---")
    return run_training_trial(params, trial_num=trial.number)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced DNA CNN Training with Hyperparameter Tuning")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna tuning trials.")
    args = parser.parse_args()

    if os.path.exists("results"): shutil.rmtree("results")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials, timeout=7200)
    
    print("\n\n" + "="*80 + "\n" + " " * 25 + "HYPERPARAMETER TUNING COMPLETE\n" + "="*80)
    
    best_trial = study.best_trial
    print(f"Total trials: {len(study.trials)}\nBest trial number: {best_trial.number}\nBest Validation Accuracy: {best_trial.value:.4f}\n")
    print("--- Optimal Hyperparameters ---"); [print(f"{k:>15}: {v}") for k, v in best_trial.params.items()]; print("-" * 33)

    print("\nSaving artifacts from the best trial to 'results/best_trial/'...")
    best_params = best_trial.params.copy()
    best_params.update({'csv': "data/processed/ProSeq_with_5component_analysis.csv", 'epochs': 200, 'seq_len': 600, 'rc_tta': True})
    run_training_trial(best_params, trial_num="best_final", is_best_trial=True)
    