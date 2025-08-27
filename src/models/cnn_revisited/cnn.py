#!/usr/bin/env python3
"""
ProSeq CNN Classifier (v2 – fixes RecursionError by removing AUC)
Train a 1D CNN on DNA sequences (column: 'ProSeq') to predict 'Predicted_Component_5'.

Usage:
  python proseq_cnn_classifier.py --csv ProSeq_with_5component_analysis.csv --outdir ./proseq_cnn_run
"""

import argparse, json, os, random, sys, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

VALID_CHARS = {'A','C','G','T','N'}
ALPHABET = ['A','C','G','T','N']
CHAR_TO_IDX = {c:i for i,c in enumerate(ALPHABET)}

def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN to classify Predicted_Component_5 from ProSeq DNA sequences.")
    p.add_argument("--csv", default="data/processed/ProSeq_with_5component_analysis.csv")
    p.add_argument("--outdir", default="src/models/cnn_revisited/proseq_cnn_run")
    p.add_argument("--maxlen", type=int, default=None)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--kernel_sizes", type=str, default="7,11,15")
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--class_weight", action="store_true")
    return p.parse_args()

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

def build_cnn(input_len: int, num_classes: int, filters: int = 128,
              kernel_sizes=(7,11,15), dropout=0.3, lr=1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, len(ALPHABET)), name="onehot_input")
    convs = []
    for k in kernel_sizes:
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        convs.append(x)
    x = layers.Concatenate()(convs) if len(convs) > 1 else convs[0]
    x = layers.Conv1D(filters*2, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="out")(x)
        loss = "binary_crossentropy"
    else:
        out = layers.Dense(num_classes, activation="softmax", name="out")(x)
        loss = "sparse_categorical_crossentropy"
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=loss,
                  metrics=["accuracy"])  # AUC removed to avoid recursion errors
    return model

def plot_history(history: tf.keras.callbacks.History, out_png: str):
    hist = pd.DataFrame(history.history)
    hist.to_csv(out_png.replace(".png", ".csv"), index=False)
    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist.columns:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def main():
    print('Starting CNN training...')
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    df = df.dropna(subset=["ProSeq","Predicted_Component_5"]).copy()
    df["ProSeq"] = df["ProSeq"].astype(str).map(clean_seq)

    seq_lens = df["ProSeq"].str.len().values
    maxlen = int(np.percentile(seq_lens, 95)) if args.maxlen is None else args.maxlen
    maxlen = max(50, min(maxlen, int(np.max(seq_lens))))

    le = LabelEncoder()
    y = le.fit_transform(df["Predicted_Component_5"].astype(str).values)
    classes = list(le.classes_)
    num_classes = len(classes)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df["ProSeq"].values, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    val_frac_of_train = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac_of_train, random_state=SEED, stratify=y_train_val
    )

    X_train_oh = sequences_to_onehot(X_train, maxlen)
    X_val_oh   = sequences_to_onehot(X_val,   maxlen)
    X_test_oh  = sequences_to_onehot(X_test,  maxlen)

    class_weight = None
    if args.class_weight:
        uniq = np.unique(y_train)
        cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y_train)
        class_weight = {int(k): float(v) for k, v in zip(uniq, cw)}

    kernel_sizes = tuple(int(k.strip()) for k in args.kernel_sizes.split(","))
    model = build_cnn(maxlen, num_classes, filters=args.filters,
                      kernel_sizes=kernel_sizes, dropout=args.dropout, lr=args.learning_rate)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(args.outdir, "model.h5"), monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.early_stop//2), min_lr=1e-6),
    ]

    history = model.fit(
        X_train_oh, y_train,
        validation_data=(X_val_oh, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=2
    )

    plot_history(history, os.path.join(args.outdir, "training_loss.png"))

    model = tf.keras.models.load_model(os.path.join(args.outdir, "model.h5"))
    y_prob = model.predict(X_test_oh, batch_size=args.batch_size)
    y_pred = (y_prob.ravel() >= 0.5).astype(int) if num_classes == 2 else np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)

    with open(os.path.join(args.outdir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)
    with open(os.path.join(args.outdir, "class_report.txt"), "w") as f:
        f.write(report)

    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar(im)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png")); plt.close()

    with open(os.path.join(args.outdir, "tokenizer.json"), "w") as f:
        json.dump({"alphabet": ALPHABET, "char_to_idx": CHAR_TO_IDX, "maxlen": maxlen}, f, indent=2)

    print("Done.")
    print(f"Test accuracy: {acc:.4f}")
    print("Classes:", classes)

if __name__ == "__main__":
    main()
