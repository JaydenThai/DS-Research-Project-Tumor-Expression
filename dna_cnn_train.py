#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA CNN (CUDA/MPS/CPU) for ~8k samples
- Compact CNN for motif discovery (valid for 600bp DNA)
- Robust cleaning & IUPAC handling
- Losses: ce | focal | balanced_ce | ldam | logit_adj
- Cosine LR with warmup, AdamW, grad clip
- Reverse-complement TTA on validation for mono encoding
- Optional mild MixUp

Outputs: model_final.pt, model_best.pt, metrics.csv, plots/
"""

import argparse, os, random, math
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------- Device ----------------
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ---------------- IUPAC & helpers ----------------
IUPAC = {
    "A":[1,0,0,0], "C":[0,1,0,0], "G":[0,0,1,0], "T":[0,0,0,1],
    "R":[0.5,0,0.5,0], "Y":[0,0.5,0,0.5], "S":[0,0.5,0.5,0],
    "W":[0.5,0,0,0.5], "K":[0,0,0.5,0.5], "M":[0.5,0.5,0,0],
    "B":[0,1/3,1/3,1/3], "D":[1/3,0,1/3,1/3], "H":[1/3,1/3,0,1/3],
    "V":[1/3,1/3,1/3,0], "N":[0.25,0.25,0.25,0.25]
}
IUPAC_KEYS = set(IUPAC.keys())
RC_TABLE = str.maketrans({
    "A":"T","C":"G","G":"C","T":"A",
    "R":"Y","Y":"R","S":"S","W":"W","K":"M","M":"K",
    "B":"V","V":"B","D":"H","H":"D","N":"N"
})

def clean_seq(seq: object, L: int) -> str:
    if not isinstance(seq, str):
        if seq is None: s = ""
        elif isinstance(seq, float) and (math.isnan(seq) if hasattr(math, "isnan") else False): s = ""
        else: s = str(seq)
    else:
        s = seq
    s = s.upper()
    s = "".join(ch if ch in IUPAC_KEYS else "N" for ch in s)
    return s[:L] if len(s) >= L else s + ("N" * (L - len(s)))

def rev_comp_str(seq: str) -> str:
    s = "".join(ch if ch in IUPAC_KEYS else "N" for ch in seq.upper())
    return s.translate(RC_TABLE)[::-1]

def mutate(seq: str, rate: float = 0.01) -> str:
    if rate <= 0: return seq
    bases = ["A","C","G","T"]
    seq = list(seq)
    for i,ch in enumerate(seq):
        if ch in bases and random.random() < rate:
            seq[i] = random.choice([b for b in bases if b != ch])
    return "".join(seq)

def one_hot_mono(seq: str, L: int) -> np.ndarray:
    arr = np.zeros((4,L), dtype=np.float32)
    upto = min(L, len(seq))
    for i in range(upto):
        arr[:, i] = IUPAC.get(seq[i], IUPAC["N"])
    return arr

def one_hot_dinuc(seq: str, L: int) -> np.ndarray:
    arr = np.zeros((16,L), dtype=np.float32)
    idx = {"A":0,"C":1,"G":2,"T":3}
    upto = min(L-1, len(seq)-1)
    for i in range(upto):
        a,b = seq[i],seq[i+1]
        if a in idx and b in idx:
            arr[idx[a]*4+idx[b], i] = 1.0
    return arr

# ---------------- Dataset ----------------
class DNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=600, encoding="mono",
                 augment=False, mutate_rate=0.01, rc_prob=0.5):
        self.seq_len=seq_len; self.encoding=encoding
        self.augment=augment; self.mutate_rate=mutate_rate; self.rc_prob=rc_prob

        df = df.copy()
        df["ProSeq"] = df["ProSeq"].apply(lambda x: clean_seq(x, self.seq_len))
        if "Predicted_Component_5" in df.columns:
            labels = df["Predicted_Component_5"].astype(int).values - 1
        else:
            prob_cols = [c for c in df.columns if c.endswith("_Probability")]
            if not prob_cols: raise ValueError("Need Predicted_Component_5 or *_Probability.")
            labels = df[prob_cols].values.argmax(axis=1)
        self.labels = labels.astype(int)
        self.seqs = df["ProSeq"].tolist()

    def __len__(self): return len(self.seqs)

    def _encode(self, s: str) -> torch.Tensor:
        if self.encoding=="mono":
            arr = one_hot_mono(s, self.seq_len)
        else:
            arr = one_hot_dinuc(s, self.seq_len)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return torch.as_tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.seqs[idx]; y = int(self.labels[idx])
        if self.augment:
            if random.random() < self.rc_prob: s = rev_comp_str(s)
            s = mutate(s, self.mutate_rate)
        x = self._encode(s)
        return x, y

# ---------------- Model ----------------
class SmallDNACNN(nn.Module):
    """Compact CNN sized for ~8k samples and 600bp."""
    def __init__(self, in_ch: int, n_classes: int, p_drop: float=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 15, padding=7)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 96, 7, padding=3)
        self.bn2   = nn.BatchNorm1d(96)
        self.conv3 = nn.Conv1d(96, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(p_drop)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

# ---------------- Losses ----------------
class BalancedSoftmaxCE(nn.Module):
    """Ren et al. Balanced Softmax: softmax(logits + log N_c)."""
    def __init__(self, cls_counts: torch.Tensor):
        super().__init__()
        self.register_buffer("log_counts", torch.log(torch.clamp(cls_counts.float(), min=1.0)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logits = logits + self.log_counts.unsqueeze(0)  # broadcast
        return F.cross_entropy(logits, target)

class LDAMLoss(nn.Module):
    """Cao et al. LDAM-DRW margin loss (without DRW by default)."""
    def __init__(self, cls_counts: torch.Tensor, max_m: float=0.5, scale: float=30.0):
        super().__init__()
        n = cls_counts.float().clamp(min=1.0)
        m_list = 1.0 / torch.sqrt(torch.sqrt(n))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = scale

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, target.view(-1,1), 1)
        batch_m = self.m_list[target]             # (B,)
        adjusted = logits.clone()
        adjusted[index.bool()] -= batch_m
        return F.cross_entropy(self.s * adjusted, target)

class LogitAdjustedCE(nn.Module):
    """Menon et al. Logit-adjusted CE: logits + tau * log p(y)."""
    def __init__(self, cls_counts: torch.Tensor, tau: float=1.0):
        super().__init__()
        p = torch.clamp(cls_counts.float(), min=1.0)
        p = p / p.sum()
        self.register_buffer("bias", tau * torch.log(p))
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(logits + self.bias.unsqueeze(0), target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

# ---------------- MixUp (optional) ----------------
def mixup(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.2):
    if alpha <= 0: return x, F.one_hot(y, num_classes).float(), 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y_one = F.one_hot(y, num_classes).float()
    y_mixed = lam * y_one + (1 - lam) * y_one[perm]
    return x_mixed, y_mixed, lam

def soft_ce(logits: torch.Tensor, y_soft: torch.Tensor):
    logp = F.log_softmax(logits, dim=1)
    return -(y_soft * logp).sum(dim=1).mean()

# ---------------- RC TTA for mono ----------------
def rc_onehot_mono(x: torch.Tensor) -> torch.Tensor:
    """x: (B,4,L) -> reverse complement in one-hot mono space."""
    # swap A<->T (0<->3), C<->G (1<->2), then flip length
    x_rc = x.clone()
    x_rc[:,0,:], x_rc[:,3,:] = x[:,3,:], x[:,0,:]
    x_rc[:,1,:], x_rc[:,2,:] = x[:,2,:], x[:,1,:]
    x_rc = torch.flip(x_rc, dims=[2])
    return x_rc

# ---------------- Train/Eval ----------------
def train_eval(args):
    # Repro
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = get_device()
    print(f"Using {device}")

    # Load & stratified split
    df = pd.read_csv(args.csv)
    if "Predicted_Component_5" in df.columns:
        labels_all = df["Predicted_Component_5"].astype(int).values - 1
    else:
        prob_cols = [c for c in df.columns if c.endswith("_Probability")]
        if not prob_cols: raise ValueError("Need Predicted_Component_5 or *_Probability.")
        labels_all = df[prob_cols].values.argmax(axis=1)

    df_tr, df_va = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=labels_all)

    train_ds = DNADataset(df_tr, seq_len=args.seq_len, encoding=args.encoding,
                          augment=args.augment, mutate_rate=args.mutate_rate, rc_prob=args.rc_prob)
    val_ds   = DNADataset(df_va, seq_len=args.seq_len, encoding=args.encoding,
                          augment=False, mutate_rate=0.0, rc_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    in_ch = 4 if args.encoding=="mono" else 16
    n_classes = int(max(train_ds.labels.max(), val_ds.labels.max()) + 1)

    # Class counts from train set
    cls_counts = torch.tensor(np.bincount(train_ds.labels, minlength=n_classes), dtype=torch.float32)

    # Model & optim
    model = SmallDNACNN(in_ch, n_classes, p_drop=0.4).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine schedule
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Select loss
    loss_name = args.loss.lower()
    if loss_name == "balanced_ce":
        criterion = BalancedSoftmaxCE(cls_counts.to(device))
    elif loss_name == "ldam":
        criterion = LDAMLoss(cls_counts.to(device), max_m=0.5, scale=30.0)
    elif loss_name == "logit_adj":
        criterion = LogitAdjustedCE(cls_counts.to(device), tau=1.0)
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=None, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # History
    hist = {"train_loss":[],"val_loss":[],"val_acc":[],"val_f1":[]}
    best_acc = 0.0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Mild MixUp with probability pairing_prob
            if args.pairing_prob > 0 and random.random() < args.pairing_prob:
                xm, y_soft, _ = mixup(xb, yb, n_classes, alpha=0.2)
                logits = model(xm)
                loss = soft_ce(logits, y_soft) if loss_name in {"balanced_ce","ldam","logit_adj"} else soft_ce(logits, y_soft)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step(); global_step += 1

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validate (RC TTA for mono) ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                if args.encoding == "mono" and args.rc_tta:
                    logits_f = model(xb)
                    logits_r = model(rc_onehot_mono(xb))
                    logits = 0.5 * (logits_f + logits_r)
                else:
                    logits = model(xb)
                # compute loss with hard labels
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(1)
                correct += (preds == yb).sum().item(); total += yb.size(0)
                all_preds.extend(preds.cpu().tolist()); all_labels.extend(yb.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        acc = correct/total if total else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro") if total else 0.0

        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
        hist["val_acc"].append(acc); hist["val_f1"].append(f1)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")

    # Save final + metrics
    torch.save(model.state_dict(), "model_final.pt")
    pd.DataFrame(hist).to_csv("metrics.csv", index=False)

    # Plots
    os.makedirs("plots", exist_ok=True)
    plt.plot(hist["train_loss"], label="train"); plt.plot(hist["val_loss"], label="val")
    plt.legend(); plt.title("Loss"); plt.savefig("plots/loss.png"); plt.close()
    plt.plot(hist["val_acc"]); plt.title("Val Accuracy"); plt.savefig("plots/acc.png"); plt.close()
    plt.plot([1-a for a in hist["val_acc"]]); plt.title("Error Rate"); plt.savefig("plots/error.png"); plt.close()
    plt.plot(hist["val_f1"]); plt.title("Macro F1"); plt.savefig("plots/f1.png"); plt.close()

    if len(all_labels) and len(all_preds):
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
        disp = ConfusionMatrixDisplay(cm); disp.plot()
        plt.title("Confusion Matrix (val)"); plt.savefig("plots/confusion_matrix.png"); plt.close()

    print(f"[DONE] best_val_acc={best_acc:.4f}  metrics.csv + plots/ saved.")

# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/processed/ProSeq_with_5component_analysis.csv",
                   help="Path to CSV with columns: ProSeq, Predicted_Component_5")
    p.add_argument("--epochs", type=int, default=200)             # less aggressive schedule, but long training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=600, help="Pad/truncate length")
    p.add_argument("--encoding", type=str, default="mono", choices=["mono", "dinuc"])
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--mutate-rate", type=float, default=0.01)
    p.add_argument("--rc-prob", type=float, default=0.5)
    p.add_argument("--pairing-prob", type=float, default=0.15)    # mild MixUp prob
    p.add_argument("--loss", type=str, default="balanced_ce",
                   choices=["ce","focal","balanced_ce","ldam","logit_adj"])
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-4)              # gentler LR
    p.add_argument("--weight-decay", type=float, default=5e-4)    # moderate regularization
    p.add_argument("--rc-tta", action="store_true", default=True, help="Reverse-complement TTA (mono only)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_eval(args)
