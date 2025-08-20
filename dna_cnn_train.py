import argparse, os, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------- Device ----------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------- IUPAC & Encoding ----------------
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

def clean_seq(seq, L):
    """Coerce to uppercase IUPAC, replace unknowns with N, pad/truncate to L."""
    if not isinstance(seq, str):
        if seq is None: s = ""
        elif isinstance(seq, float) and math.isnan(seq): s = ""
        else: s = str(seq)
    else:
        s = seq
    s = s.upper()
    s = "".join(ch if ch in IUPAC_KEYS else "N" for ch in s)
    if len(s) >= L: return s[:L]
    return s + ("N" * (L - len(s)))

def rev_comp(seq):
    if not isinstance(seq, str):
        if seq is None: s = ""
        elif isinstance(seq, float) and math.isnan(seq): s = ""
        else: s = str(seq)
    else:
        s = seq
    s = s.upper()
    s = "".join(ch if ch in RC_TABLE or ch in IUPAC_KEYS else "N" for ch in s)
    return s.translate(RC_TABLE)[::-1]

def one_hot_mono(seq, L):
    arr = np.zeros((4, L), dtype=np.float32)
    upto = min(L, len(seq))
    for i in range(upto):
        v = IUPAC.get(seq[i], IUPAC["N"])
        arr[:, i] = v
    return arr

def one_hot_dinuc(seq, L):
    # strict dinucleotide for A/C/G/T; ambiguous pairs → zeros (safer)
    arr = np.zeros((16, L), dtype=np.float32)
    idx = {"A":0, "C":1, "G":2, "T":3}
    upto = min(L-1, len(seq)-1)
    for i in range(upto):
        a, b = seq[i], seq[i+1]
        if a in idx and b in idx:
            arr[idx[a]*4 + idx[b], i] = 1.0
    return arr

def mutate(seq, rate=0.01):
    if rate <= 0: return seq
    bases = ["A","C","G","T"]
    seq = list(seq)
    for i,ch in enumerate(seq):
        if ch in bases and random.random() < rate:
            seq[i] = random.choice([b for b in bases if b != ch])
    return "".join(seq)

# ---------------- Dataset ----------------
class DNADataset(Dataset):
    def __init__(self, df, seq_len=600, encoding="mono",
                 augment=False, mutate_rate=0.01, rc_prob=0.5, pairing_prob=0.0):
        self.seq_len = seq_len
        self.encoding = encoding
        self.augment = augment
        self.mutate_rate = mutate_rate
        self.rc_prob = rc_prob
        self.pairing_prob = pairing_prob

        # Clean upfront so everything is string & same length
        df = df.copy()
        df["ProSeq"] = df["ProSeq"].apply(lambda x: clean_seq(x, self.seq_len))
        if "Predicted_Component_5" in df.columns:
            labels = df["Predicted_Component_5"].astype(int).values - 1
        else:
            prob_cols = [c for c in df.columns if c.endswith("_Probability")]
            if not prob_cols:
                raise ValueError("Need 'Predicted_Component_5' or *_Probability columns.")
            labels = df[prob_cols].values.argmax(axis=1)
        self.labels = labels.astype(int)
        self.seqs = df["ProSeq"].tolist()

    def __len__(self): return len(self.seqs)

    def _encode(self, s):
        if self.encoding == "mono":
            arr = one_hot_mono(s, self.seq_len)
        else:
            arr = one_hot_dinuc(s, self.seq_len)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return torch.as_tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        y = int(self.labels[idx])

        if self.augment:
            if random.random() < self.rc_prob:
                s = rev_comp(s)
            s = mutate(s, self.mutate_rate)

        x = self._encode(s)

        if self.augment and self.pairing_prob > 0 and random.random() < self.pairing_prob:
            j = random.randrange(len(self.seqs))
            s2 = self.seqs[j]
            if random.random() < 0.5:
                s2 = rev_comp(s2)
            s2 = mutate(s2, self.mutate_rate)
            x2 = self._encode(s2)
            alpha = random.uniform(0.3, 0.7)
            x = alpha * x + (1 - alpha) * x2

        return x, y

# ---------------- Model ----------------
class CNNClassifier(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 15, padding=7)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 7, padding=3)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.drop  = nn.Dropout(0.5)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.fc    = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

# ---------------- Train/Eval ----------------
def train_eval(args):
    device = get_device()
    print(f"Using {device}")

    df = pd.read_csv(args.csv)
    ds = DNADataset(
        df,
        seq_len=args.seq_len,
        encoding=args.encoding,
        augment=args.augment,
        mutate_rate=args.mutate_rate,
        rc_prob=args.rc_prob,
        pairing_prob=args.pairing_prob
    )
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_trn = n - n_val
    train_ds, val_ds = random_split(ds, [n_trn, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    in_ch = 4 if args.encoding == "mono" else 16
    n_classes = int(max(ds.labels) + 1)
    model = CNNClassifier(in_ch, n_classes).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        acc = correct / total if total else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro") if total else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")

    # Save final model and metrics
    torch.save(model.state_dict(), "model_final.pt")
    pd.DataFrame(history).to_csv("metrics.csv", index=False)

    # Plots
    os.makedirs("plots", exist_ok=True)
    plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val")
    plt.legend(); plt.title("Loss"); plt.savefig("plots/loss.png"); plt.close()

    plt.plot(history["val_acc"]); plt.title("Val Accuracy"); plt.savefig("plots/acc.png"); plt.close()
    plt.plot([1 - a for a in history["val_acc"]]); plt.title("Error Rate"); plt.savefig("plots/error.png"); plt.close()
    plt.plot(history["val_f1"]); plt.title("Macro F1"); plt.savefig("plots/f1.png"); plt.close()

    if len(all_labels) and len(all_preds):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Confusion Matrix (val)")
        plt.savefig("plots/confusion_matrix.png")
        plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/processed/ProSeq_with_5component_analysis.csv",
                   help="Path to CSV with columns: ProSeq, Predicted_Component_5")
    p.add_argument("--epochs", type=int, default=200)                 # increased
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=600, help="Pad/truncate length")
    p.add_argument("--encoding", type=str, default="mono", choices=["mono", "dinuc"])
    p.add_argument("--augment", action="store_true", default=True)    # default True
    p.add_argument("--mutate-rate", type=float, default=0.015)
    p.add_argument("--rc-prob", type=float, default=0.5)
    p.add_argument("--pairing-prob", type=float, default=0.3)
    p.add_argument("--loss", type=str, default="hybrid",
                   choices=["crossentropy", "focal", "cosine_ce", "triplet", "hybrid"])
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-3)                  # increased
    p.add_argument("--weight-decay", type=float, default=1e-3)        # stronger regularization
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    train_eval(args)
