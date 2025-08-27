
import argparse, json, os, random, sys, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 1356924
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

VALID_CHARS = {'A','C','G','T','N'}
ALPHABET = ['A','C','G','T','N']
CHAR_TO_IDX = {c:i for i,c in enumerate(ALPHABET)}

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ordinal (CORAL) CNN for Predicted_Component_5 from ProSeq DNA.")
    p.add_argument("--csv", default="data/processed/ProSeq_with_5component_analysis.csv")
    p.add_argument("--outdir", default="./proseq_ordinal_run")
    p.add_argument("--maxlen", type=int, default=None, help="If unset, uses 95th percentile of sequence length.")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--early_stop", type=int, default=6)
    p.add_argument("--class_weight", action="store_true", help="Balance classes via per-sample weights.")
    return p.parse_args()

# -------------------- Data utils --------------------
def clean_seq(seq: str) -> str:
    s = (seq or "").upper()
    return "".join(ch if ch in VALID_CHARS else 'N' for ch in s)

def sequences_to_onehot(seqs, maxlen: int) -> np.ndarray:
    X = np.zeros((len(seqs), maxlen, len(ALPHABET)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = clean_seq(s)[:maxlen]
        for j, ch in enumerate(s):
            X[i, j, CHAR_TO_IDX.get(ch, 4)] = 1.0
    return X

# -------------------- Ordinal (CORAL) helpers --------------------
def to_coral_targets(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build cumulative targets: t[:, k] = 1 if y >= k+1 else 0, for k in 0..K-2
    y_int: [N] with values in [0..K-1]
    returns: [N, K-1] float32
    """
    N = len(y_int); K = num_classes
    t = np.zeros((N, K-1), dtype=np.float32)
    for k in range(K-1):
        t[:, k] = (y_int >= (k+1)).astype(np.float32)
    return t

def coral_pred_to_label(p: np.ndarray) -> np.ndarray:
    """Convert sigmoid thresholds to ordinal class: count thresholds >= 0.5."""
    return (p >= 0.5).sum(axis=1).astype(np.int32)

def expected_class_from_coral(p: np.ndarray) -> np.ndarray:
    """Expected class index (float) as sum of threshold probabilities."""
    return p.sum(axis=1)

# -------------------- Model --------------------
def build_backbone(input_len: int, filters: int = 128, dropout: float = 0.35) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, len(ALPHABET)))
    # Stem
    x = layers.Conv1D(filters, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Dilated residual stack (captures longer motifs efficiently)
    def res_block(z, f, k, d):
        y = layers.Conv1D(f, k, padding="same", dilation_rate=d, activation="relu")(z)
        y = layers.BatchNormalization()(y)
        y = layers.Conv1D(f, k, padding="same", dilation_rate=d, activation="relu")(y)
        y = layers.BatchNormalization()(y)
        if z.shape[-1] != f:
            z = layers.Conv1D(f, 1, padding="same")(z)
        out = layers.Add()([z, y])
        out = layers.Activation("relu")(out)
        out = layers.MaxPooling1D(2)(out)
        return out

    x = res_block(x, filters,   7, 1)
    x = res_block(x, filters*2, 7, 2)
    x = res_block(x, filters*2, 7, 4)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return models.Model(inp, x, name="dna_backbone")

def build_coral(input_len: int, num_classes: int, lr: float, filters: int, dropout: float) -> tf.keras.Model:
    """
    CORAL head: Dense(K-1, sigmoid) estimating P(y >= k), k=1..K-1
    Train with average BinaryCrossentropy across thresholds.
    """
    bb = build_backbone(input_len, filters, dropout)
    inp = bb.input
    x = bb.output
    out = layers.Dense(num_classes - 1, activation="sigmoid", name="coral_out")(x)
    model = models.Model(inp, out, name="coral_model")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.BinaryCrossentropy(),  # averaged over columns
                  metrics=[])  # custom eval later
    return model

# -------------------- Training helpers --------------------
def plot_history(history: tf.keras.callbacks.History, out_png: str):
    hist = pd.DataFrame(history.history)
    hist.to_csv(out_png.replace(".png", ".csv"), index=False)
    plt.figure()
    for k, v in hist.items():
        if not k.startswith("val_"):
            plt.plot(v, label=k)
    for k, v in hist.items():
        if k.startswith("val_"):
            plt.plot(v, label=k)
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    need = {"ProSeq", "Predicted_Component_5"}
    if not need.issubset(df.columns):
        print("ERROR: CSV must contain 'ProSeq' and 'Predicted_Component_5'.", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=["ProSeq","Predicted_Component_5"]).copy()
    df["ProSeq"] = df["ProSeq"].astype(str).map(clean_seq)

    # Sequence length policy
    seq_lens = df["ProSeq"].str.len().values
    if args.maxlen is None:
        maxlen = int(np.percentile(seq_lens, 95))
        maxlen = max(50, min(maxlen, int(np.max(seq_lens))))
    else:
        maxlen = args.maxlen

    # Encode labels (NOTE: order is by LabelEncoder; see classes_order.txt)
    le = LabelEncoder()
    y = le.fit_transform(df["Predicted_Component_5"].astype(str).values)
    classes = list(le.classes_)
    K = len(classes)
    if K < 2:
        print("ERROR: Need at least 2 unique classes.", file=sys.stderr)
        sys.exit(1)

    # Splits (stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df["ProSeq"].values, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    val_frac_of_train = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac_of_train, random_state=SEED, stratify=y_train_val
    )

    # One-hot DNA
    X_train_oh = sequences_to_onehot(X_train, maxlen)
    X_val_oh   = sequences_to_onehot(X_val,   maxlen)
    X_test_oh  = sequences_to_onehot(X_test,  maxlen)

    # CORAL targets
    y_train_coral = to_coral_targets(y_train, K)
    y_val_coral   = to_coral_targets(y_val,   K)

    # Optional class weighting -> per-sample weights (no one-hot labels here)
    sample_weight = None
    if args.class_weight:
        uniq = np.unique(y_train)
        cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y_train)
        class_weight = {int(k): float(v) for k, v in zip(uniq, cw)}
        sw = np.vectorize(class_weight.get)(y_train)  # [N_train]
        sample_weight = sw

    # Build + train
    model = build_coral(maxlen, K, args.learning_rate, args.filters, args.dropout)
    monitor, mode = "val_loss", "min"
    cbs = [
        callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=args.early_stop, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(args.outdir, "model.h5"), monitor=monitor, mode=mode, save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5, patience=max(2, args.early_stop//2), min_lr=1e-6),
    ]

    history = model.fit(
        X_train_oh, y_train_coral,
        validation_data=(X_val_oh, y_val_coral),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        sample_weight=sample_weight,
        verbose=2
    )

    plot_history(history, os.path.join(args.outdir, "training_metrics.png"))

    # Evaluate
    model = tf.keras.models.load_model(os.path.join(args.outdir, "model.h5"))
    y_thr = model.predict(X_test_oh, batch_size=args.batch_size)  # [N, K-1]
    y_pred = coral_pred_to_label(y_thr)
    y_exp  = expected_class_from_coral(y_thr)

    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_exp = float(np.mean(np.abs(y_exp - y_test)))

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)

    # Save artefacts
    with open(os.path.join(args.outdir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": float(acc),
            "mae": float(mae),
            "mae_expected_class": float(mae_exp)
        }, f, indent=2)

    with open(os.path.join(args.outdir, "class_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar(im)
    ticks = np.arange(K)
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png")); plt.close()

    # Save tokenizer + maxlen
    with open(os.path.join(args.outdir, "tokenizer.json"), "w") as f:
        json.dump({"alphabet": ALPHABET, "char_to_idx": CHAR_TO_IDX, "maxlen": int(maxlen)}, f, indent=2)

    # Save class order used (important for ordinal meaning)
    with open(os.path.join(args.outdir, "classes_order.txt"), "w") as f:
        f.write("\n".join(f"{i}: {c}" for i, c in enumerate(classes)))

    print("Done.")
    print(f"Accuracy: {acc:.4f}  MAE: {mae:.4f}  Expected-class MAE: {mae_exp:.4f}")
    print("Classes (encoder order):", classes)

if __name__ == "__main__":
    main()
