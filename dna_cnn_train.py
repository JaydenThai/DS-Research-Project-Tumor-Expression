
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA CNN Trainer (CUDA & MPS compatible)

Features for low-data regimes (~8k samples):
- One-hot ("mono") or dinucleotide ("dinuc") encoding
- Reverse-complement augmentation
- Low-rate random substitution mutations
- Optional SamplePairing (convex mixing of two one-hots)
- Multiscale CNN with global average pooling
- Multiple loss options:
    * crossentropy (label smoothing)
    * focal
    * cosine_ce (cosine-softmax head)
    * triplet (metric learning only)
    * hybrid (crossentropy + triplet)
- Early stopping on validation accuracy
- Saves best model to model_best.pt
- Works with CUDA, MPS, or CPU

Example:
    python dna_cnn_train.py --csv /mnt/data/ProSeq_with_5component_analysis.csv --epochs 30 --loss hybrid --augment --pairing-prob 0.3
"""
import argparse
import os
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --------------- Device selection (CUDA, MPS, CPU) ---------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# --------------- Utilities ---------------
NUC = "ACGT"
NUC_SET = set(NUC)
DINUC = [a+b for a in NUC for b in NUC]
DINUC_INDEX = {d:i for i, d in enumerate(DINUC)}

def fix_length(seq: str, length: int) -> str:
    """Pad with 'N' or truncate to target length."""
    s = (seq or "").upper()
    if len(s) >= length:
        return s[:length]
    return s + ("N" * (length - len(s)))

def reverse_complement(seq: str) -> str:
    comp = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))

def mutate_seq(seq: str, mutate_rate: float=0.01) -> str:
    if mutate_rate <= 0:
        return seq
    seq_list = list(seq)
    for i, ch in enumerate(seq_list):
        if random.random() < mutate_rate:
            seq_list[i] = random.choice("ACGT")
    return "".join(seq_list)

def one_hot_mono(seq: str) -> np.ndarray:
    # Returns shape (C=4, L)
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    mapping = {"A":0, "C":1, "G":2, "T":3}
    for i, ch in enumerate(seq):
        idx = mapping.get(ch, None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

def one_hot_dinuc(seq: str) -> np.ndarray:
    # Returns shape (C=16, L-1) but we'll keep output length L by padding one on left
    L = len(seq)
    arr = np.zeros((16, L), dtype=np.float32)
    for i in range(L-1):
        pair = seq[i:i+2]
        if len(pair)==2 and all(c in NUC_SET for c in pair):
            j = DINUC_INDEX[pair]
            arr[j, i+1] = 1.0
    return arr

def samplepair_mix(enc1: torch.Tensor, enc2: torch.Tensor, alpha: float=0.5) -> torch.Tensor:
    # enc1/enc2: (C, L)
    return alpha*enc1 + (1.0 - alpha)*enc2

# --------------- Dataset ---------------
class DNADataset(torch.utils.data.Dataset):
    """
    DNA dataset with explicit alphabet-based encoders.

    - encoding="mono": 4 channels (A,C,G,T) with IUPAC ambiguous codes mapped to
      uniform soft vectors over their possibilities (e.g., N -> [0.25,0.25,0.25,0.25]).
    - encoding="dinuc": 16 channels (AA..TT). Ambiguous pairs map to zeros by default
      (simpler & safer for small data); you can extend this to soft pair encodings later.
    """
    def __init__(self,
                 seqs: List[str],
                 labels: List[int],
                 seq_len: int = 600,
                 encoding: str = "mono",
                 augment: bool = False,
                 mutate_rate: float = 0.01,
                 rc_prob: float = 0.5,
                 pairing_prob: float = 0.0):
        super().__init__()
        assert encoding in ("mono", "dinuc")
        self.seqs = seqs
        self.labels = labels
        self.seq_len = seq_len
        self.encoding = encoding
        self.augment = augment
        self.mutate_rate = mutate_rate
        self.rc_prob = rc_prob
        self.pairing_prob = pairing_prob

        # --- Build alphabets ---
        # Mono alphabet (IUPAC soft encodings)
        self.alphabet = self._build_mono_alphabet()      # dict[str -> np.ndarray(4,)]
        self.unk_vec_mono = np.zeros((4,), dtype=np.float32)

        # Dinucleotide alphabet (strict AC/CG/.. only)
        nucs = ["A", "C", "G", "T"]
        pairs = [a + b for a in nucs for b in nucs]
        self.alphabet_dinuc = {
            p: self._one_hot_index(i, depth=16) for i, p in enumerate(pairs)
        }                                              # dict[str -> np.ndarray(16,)]
        self.unk_vec_dinuc = np.zeros((16,), dtype=np.float32)

        # Cache channel count
        self.channels = 4 if self.encoding == "mono" else 16

    def __len__(self):
        return len(self.seqs)

    # ---------- Encoding helpers ----------
    def _one_hot_index(self, idx: int, depth: int) -> np.ndarray:
        v = np.zeros((depth,), dtype=np.float32)
        v[idx] = 1.0
        return v

    def _build_mono_alphabet(self) -> dict:
        """
        IUPAC encoding for mono-nucleotides.
        Soft vectors sum to 1 across possible bases (uniform).
        """
        A = np.array([1, 0, 0, 0], dtype=np.float32)
        C = np.array([0, 1, 0, 0], dtype=np.float32)
        G = np.array([0, 0, 1, 0], dtype=np.float32)
        T = np.array([0, 0, 0, 1], dtype=np.float32)

        def u(*vecs):
            # Uniform over provided base vectors
            m = np.stack(vecs, axis=0).mean(axis=0).astype(np.float32)
            return m

        # Core & IUPAC ambiguous
        alpha = {
            "A": A, "C": C, "G": G, "T": T,
            "R": u(A, G),      # A/G
            "Y": u(C, T),      # C/T
            "S": u(C, G),      # C/G
            "W": u(A, T),      # A/T
            "K": u(G, T),      # G/T
            "M": u(A, C),      # A/C
            "B": u(C, G, T),   # not A
            "D": u(A, G, T),   # not C
            "H": u(A, C, T),   # not G
            "V": u(A, C, G),   # not T
            "N": u(A, C, G, T) # any
        }
        return alpha

    def _encode_mono(self, s: str) -> torch.Tensor:
        """
        Returns (4, L) tensor. Unknown chars -> zeros (or N’s soft vector if present).
        """
        L = self.seq_len
        x = np.zeros((4, L), dtype=np.float32)
        # Use up to seq_len characters
        upto = min(len(s), L)
        for i in range(upto):
            ch = s[i]
            vec = self.alphabet.get(ch)
            if vec is None:
                # If truly unknown (not IUPAC), choose zero vector
                vec = self.unk_vec_mono
            x[:, i] = vec
        # If shorter than L, remaining columns stay zeros (already padded upstream)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.as_tensor(x, dtype=torch.float32)

    def _encode_dinuc(self, s: str) -> torch.Tensor:
        """
        Returns (16, L) tensor with a left-padding convention:
        - channel set for pair s[i:i+2] is written at column i+1 (so column 0 is zeros).
        - Pairs containing non-ACGT characters map to zeros (unk pair).
        """
        L = self.seq_len
        x = np.zeros((16, L), dtype=np.float32)
        upto = min(len(s), L)
        # Fill pairs while staying within bounds (i+1 < L)
        for i in range(min(upto - 1, L - 1)):
            pair = s[i:i+2]
            vec = self.alphabet_dinuc.get(pair, self.unk_vec_dinuc)
            x[:, i + 1] = vec
        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.as_tensor(x, dtype=torch.float32)

    def _encode(self, s: str) -> torch.Tensor:
        if self.encoding == "mono":
            return self._encode_mono(s)
        else:
            return self._encode_dinuc(s)

    # ---------- __getitem__ with your augmentations ----------
    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        y = self.labels[idx]

        # Augment
        if self.augment:
            if random.random() < self.rc_prob:
                s = reverse_complement(s)
            s = mutate_seq(s, self.mutate_rate)

        x = self._encode(s)  # (C, L)

        # Optional SamplePairing (convex mixing)
        if self.augment and self.pairing_prob > 0.0 and random.random() < self.pairing_prob:
            j = random.randrange(len(self.seqs))
            s2 = self.seqs[j]
            if self.augment:
                if random.random() < 0.5:
                    s2 = reverse_complement(s2)
                s2 = mutate_seq(s2, self.mutate_rate)
            x2 = self._encode(s2)
            alpha = random.uniform(0.3, 0.7)
            x = alpha * x + (1.0 - alpha) * x2

        return x, torch.tensor(y, dtype=torch.long)


# --------------- Model ---------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        pad = k // 2
        self.seq = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class MultiScaleCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, embed_dim: int=128, use_cosine: bool=False):
        super().__init__()
        self.use_cosine = use_cosine

        ks = [8, 16, 32]
        branches = []
        for _k in ks:
            branches.append(ConvBlock(in_ch, 64, _k))
        self.branches = nn.ModuleList(branches)

        self.merge_conv = nn.Conv1d(64*len(ks), 128, kernel_size=3, padding=1, bias=False)
        self.merge_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

        self.embed = nn.Linear(128, embed_dim)  # after GAP to (B, 128)
        self.embed_bn = nn.BatchNorm1d(embed_dim)

        if not use_cosine:
            self.classifier = nn.Linear(embed_dim, n_classes)
        else:
            # Cosine-softmax head: normalize weights; we scale logits by s
            self.classifier = CosineClassifier(embed_dim, n_classes, s=30.0)

    def forward(self, x):
        # x: (B, C, L)
        outs = []
        for b in self.branches:
            y = b(x)                 # (B, 64, L)
            y = F.max_pool1d(y, 4)   # (B, 64, L/4)
            outs.append(y)
        z = torch.cat(outs, dim=1)   # (B, 64*3, L/4)
        z = F.relu(self.merge_bn(self.merge_conv(z)))  # (B,128, L/4)
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)    # (B,128)
        z = self.dropout(z)
        emb = self.embed_bn(self.embed(z))             # (B,embed_dim)
        emb = F.relu(emb)
        logits = self.classifier(emb) if hasattr(self, "classifier") else None
        return emb, logits

class CosineClassifier(nn.Module):
    def __init__(self, in_features, n_classes, s=30.0, m=0.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_features))
        nn.init.xavier_normal_(self.W)
        self.s = s
        self.m = m

    def forward(self, x):
        # x: (B, D); W: (C, D)
        x_norm = F.normalize(x, dim=1)          # (B, D)
        w_norm = F.normalize(self.W, dim=1)     # (C, D)
        cos = torch.matmul(x_norm, w_norm.t())  # (B, C)
        logits = self.s * (cos - self.m)
        return logits

# --------------- Losses ---------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # Cross-entropy with label smoothing
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    # x: (B, D)
    # returns squared Euclidean dist matrix (B,B)
    dot = torch.matmul(x, x.t())           # (B,B)
    sq = torch.diag(dot).unsqueeze(1)      # (B,1)
    dist = sq - 2*dot + sq.t()
    dist = torch.clamp(dist, min=0.0)
    return dist

def batch_semihard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float=0.2):
    """
    Semi-hard triplet loss within a batch:
        For each anchor, choose hardest positive and semi-hard negative.
    """
    B = emb.size(0)
    if B < 3:
        return emb.sum() * 0.0

    dist = pairwise_distances(emb)  # (B,B)
    labels = labels.view(-1,1)
    mask_pos = (labels == labels.t()).float()
    mask_neg = (labels != labels.t()).float()

    # For each anchor, hardest positive (max dist among positives excluding self)
    pos_dist = dist * mask_pos + (1 - mask_pos) * (-1e6)
    hardest_pos, _ = pos_dist.max(dim=1)

    # Semi-hard negatives: d(an, nn) > d(ap) but closest among those
    neg_dist = dist + (1 - mask_neg) * (1e6)
    semihard_neg = []
    for i in range(B):
        d_ap = hardest_pos[i].item()
        cand = neg_dist[i]
        mask = (cand > d_ap) & (cand < 1e6)
        if mask.any():
            semihard_neg.append(cand[mask].min())
        else:
            # fallback to hardest negative
            semihard_neg.append(neg_dist[i].min())
    semihard_neg = torch.stack(semihard_neg)

    losses = F.relu(hardest_pos - semihard_neg + margin)
    return losses.mean()

# --------------- Training / Evaluation ---------------
def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loaders(df: pd.DataFrame,
                 seq_col: str = "ProSeq",
                 label_col: str = "Predicted_Component_5",
                 seq_len: int = 600,
                 encoding: str = "mono",
                 batch_size: int = 64,
                 augment: bool = True,
                 mutate_rate: float = 0.01,
                 rc_prob: float = 0.5,
                 pairing_prob: float = 0.0,
                 val_size: float = 0.2,
                 seed: int = 42):
    # Filter rows with valid sequences & labels
    df = df.dropna(subset=[seq_col, label_col]).copy()
    df[seq_col] = df[seq_col].astype(str).str.upper().apply(lambda s: fix_length(s, seq_len))
    # Labels appear to be 1..5; convert to 0..4
    labels = df[label_col].astype(int).values - 1
    seqs = df[seq_col].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        seqs, labels, test_size=val_size, random_state=seed, stratify=labels
    )
    train_ds = DNADataset(X_train, y_train, seq_len, encoding, augment, mutate_rate, rc_prob, pairing_prob)
    val_ds   = DNADataset(X_val, y_val,   seq_len, encoding, augment=False, mutate_rate=0.0, rc_prob=0.0, pairing_prob=0.0)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, len(set(labels))

def train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_triplet=None, lambda_triplet=0.2):
    model.train()
    total_loss, total_main, total_trip = 0.0, 0.0, 0.0
    all_preds, all_tgts = [], []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        emb, logits = model(xb)
        main_loss = 0.0
        if criterion_main is not None and logits is not None:
            main_loss = criterion_main(logits, yb)

        triplet_loss = 0.0
        if criterion_triplet is not None:
            triplet_loss = criterion_triplet(emb, yb)

        loss = main_loss + lambda_triplet * triplet_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_main += (main_loss if isinstance(main_loss, float) else main_loss.item()) * xb.size(0)
        total_trip += (triplet_loss if isinstance(triplet_loss, float) else triplet_loss.item()) * xb.size(0)

        if logits is not None:
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_tgts.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(train_loader.dataset))
    avg_main = total_main / max(1, len(train_loader.dataset))
    avg_trip = total_trip / max(1, len(train_loader.dataset))
    acc = accuracy_score(all_tgts, all_preds) if all_preds else 0.0
    f1 = f1_score(all_tgts, all_preds, average='macro') if all_preds else 0.0
    return avg_loss, avg_main, avg_trip, acc, f1

@torch.no_grad()
def evaluate(model, val_loader, criterion_main, criterion_triplet=None, lambda_triplet=0.2):
    model.eval()
    total_loss, total_main, total_trip = 0.0, 0.0, 0.0
    all_preds, all_tgts = [], []

    for xb, yb in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        emb, logits = model(xb)
        main_loss = 0.0
        if criterion_main is not None and logits is not None:
            main_loss = criterion_main(logits, yb)
        triplet_loss = 0.0
        if criterion_triplet is not None:
            triplet_loss = criterion_triplet(emb, yb)
        loss = main_loss + lambda_triplet * triplet_loss

        total_loss += loss.item() * xb.size(0)
        total_main += (main_loss if isinstance(main_loss, float) else main_loss.item()) * xb.size(0)
        total_trip += (triplet_loss if isinstance(triplet_loss, float) else triplet_loss.item()) * xb.size(0)

        if logits is not None:
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_tgts.extend(yb.cpu().numpy())

    avg_loss = total_loss / max(1, len(val_loader.dataset))
    avg_main = total_main / max(1, len(val_loader.dataset))
    avg_trip = total_trip / max(1, len(val_loader.dataset))
    acc = accuracy_score(all_tgts, all_preds) if all_preds else 0.0
    f1 = f1_score(all_tgts, all_preds, average='macro') if all_preds else 0.0
    return avg_loss, avg_main, avg_trip, acc, f1

# --------------- Main ---------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/processed/ProSeq_with_5component_analysis.csv", help="Path to CSV with columns: ProSeq, Predicted_Component_5")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=600, help="Pad/truncate length")
    p.add_argument("--encoding", type=str, default="mono", choices=["mono", "dinuc"])
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--mutate-rate", type=float, default=0.015)
    p.add_argument("--rc-prob", type=float, default=0.5)
    p.add_argument("--pairing-prob", type=float, default=0.3)
    p.add_argument("--loss", type=str, default="hybrid", choices=["crossentropy", "focal", "cosine_ce", "triplet", "hybrid"])
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if "ProSeq" not in df.columns:
        raise ValueError("CSV must contain a 'ProSeq' column with DNA sequences.")
    label_col = "Predicted_Component_5" if "Predicted_Component_5" in df.columns else None
    if label_col is None:
        # Fallback: try to infer from probability columns by argmax
        prob_cols = [c for c in df.columns if c.endswith("_Probability")]
        if not prob_cols:
            raise ValueError("CSV must contain 'Predicted_Component_5' or component probability columns to infer labels.")
        print("[WARN] 'Predicted_Component_5' not found; inferring labels via argmax of *_Probability columns.")
        probs = df[prob_cols].values
        labels = probs.argmax(axis=1)  # 0..K-1
        df["__LABEL__"] = labels + 1   # mimic 'Predicted_Component_5' style (1..K)
        label_col = "__LABEL__"

    train_loader, val_loader, n_classes = make_loaders(
        df, seq_col="ProSeq", label_col=label_col, seq_len=args.seq_len, encoding=args.encoding,
        batch_size=args.batch_size, augment=args.augment,
        mutate_rate=args.mutate_rate, rc_prob=args.rc_prob, pairing_prob=args.pairing_prob if args.augment else 0.0,
        val_size=0.2, seed=args.seed
    )
    in_ch = 4 if args.encoding == "mono" else 16

    use_cosine = (args.loss == "cosine_ce")
    model = MultiScaleCNN(in_ch=in_ch, n_classes=n_classes, embed_dim=128, use_cosine=use_cosine).to(DEVICE)

    # Loss functions
    criterion_main = None
    if args.loss in ("crossentropy", "hybrid", "cosine_ce"):
        criterion_main = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        criterion_main = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
    elif args.loss == "triplet":
        criterion_main = None

    criterion_triplet = None
    if args.loss in ("triplet", "hybrid"):
        criterion_triplet = lambda emb, y: batch_semihard_triplet_loss(emb, y, margin=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    best_path = "model_best.pt"
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_main, tr_trip, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_triplet, 0.2)
        va_loss, va_main, va_trip, va_acc, va_f1 = evaluate(model, val_loader, criterion_main, criterion_triplet, 0.2)
        scheduler.step(va_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss: {tr_loss:.4f} (CE {tr_main:.4f}, Tri {tr_trip:.4f}) acc: {tr_acc:.4f} f1: {tr_f1:.4f} | "
              f"val loss: {va_loss:.4f} (CE {va_main:.4f}, Tri {va_trip:.4f}) acc: {va_acc:.4f} f1: {va_f1:.4f}")

        # Early stopping on val acc
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "val_acc": va_acc,
                        "val_f1": va_f1}, best_path)
            print(f"[INFO] New best acc {best_acc:.4f}. Saved to {best_path}.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

    print(f"[DONE] Best Val Acc: {best_acc:.4f}. Best model at {best_path}")

if __name__ == "__main__":
    main()
