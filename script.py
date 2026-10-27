"""ECG anomaly detection with an MLP autoencoder.

Usage:
    python script.py train                    # train, evaluate, save model + figures
    python script.py predict --input X.csv    # classify new ECG beats (140 values per row)
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# Config
DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
OUT_DIR = "outputs"
SEED = 42
EPOCHS = 100
BATCH = 64
LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Data: ECG5000, 140 points per beat, last column = label (1 normal, 0 anomaly)
def load_data():
    df = pd.read_csv(DATA_URL, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values == 0).astype(int)  # recode: 1 = anomaly
    return X, y


# Split 70/15/15 stratified; train set keeps normals only
def split(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_tr[y_tr == 0], (X_val, y_val), (X_te, y_te)


# Model: symmetric MLP autoencoder 140-64-32-8-32-64-140
class AutoEncoder(nn.Module):
    def __init__(self, n_in=140):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_in), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Min-max scaling with stats from training normals
def scale(X, lo, hi):
    return np.clip((X - lo) / (hi - lo), 0, 1).astype(np.float32)


# Anomaly score: per-sample mean absolute reconstruction error
@torch.no_grad()
def recon_error(model, X):
    model.eval()
    x = torch.from_numpy(X)
    return torch.mean(torch.abs(model(x) - x), dim=1).numpy()


# Training: L1 loss on normals; keep weights with best validation-normal loss
def fit(model, X_tr, X_val_norm):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    loader = torch.utils.data.DataLoader(torch.from_numpy(X_tr), batch_size=BATCH, shuffle=True)
    hist, best, best_state = {"train": [], "val": []}, np.inf, None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        val = recon_error(model, X_val_norm).mean()
        hist["train"].append(tot / len(X_tr))
        hist["val"].append(val)
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1:3d} | train {hist['train'][-1]:.4f} | val {val:.4f}")
    model.load_state_dict(best_state)
    return hist


# Threshold: maximise F1 on validation set (test set never touched)
def pick_threshold(err, y):
    cands = np.quantile(err, np.linspace(0.01, 0.99, 500))
    f1s = [f1_score(y, err > t) for t in cands]
    return float(cands[int(np.argmax(f1s))])


# Metrics on held-out test set
def evaluate(err, y, thr):
    pred = (err > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": thr,
        "auroc": roc_auc_score(y, err),
        "auprc": average_precision_score(y, err),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# Plot style: report palette, serif, thin axes, inward ticks
NAVY, RED, GOLD = "#181D61", "#D71440", "#E49F14"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9,
    "axes.linewidth": 0.6, "axes.edgecolor": "black",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.frameon": True, "legend.fancybox": False, "legend.fontsize": 7,
    "legend.edgecolor": "black", "legend.framealpha": 1, "patch.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# Blend colour with white (f = 1 full colour)
def tone(c, f):
    return tuple(1 - f * (1 - np.array(to_rgb(c))))


def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.pdf")
    plt.close(fig)


# Figures: loss curve, error histogram, ROC, confusion matrix, reconstructions
def plot_all(hist, model, X_te, y_te, err, thr):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(hist["train"], color=NAVY, lw=1, label="Train (normal)")
    ax.plot(hist["val"], color=RED, lw=1, ls="--", label="Validation (normal)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L1 loss"); ax.legend(loc="upper right")
    save(fig, "loss_curve")

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bins = np.linspace(0, err.max(), 50)
    ax.hist(err[y_te == 0], bins=bins, color=tone(NAVY, 0.75), ec="white", lw=0.3, label="Normal")
    ax.hist(err[y_te == 1], bins=bins, color=tone(RED, 0.6), ec="white", lw=0.3, label="Anomaly")
    ax.axvline(thr, color="black", lw=0.8, ls="--", label=rf"Threshold $\tau={thr:.3f}$")
    ax.set_xlabel("Reconstruction error (MAE)"); ax.set_ylabel("Count"); ax.legend(loc="upper right")
    save(fig, "error_histogram")

    fpr, tpr, _ = roc_curve(y_te, err)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot(fpr, tpr, color=NAVY, lw=1.1, label=f"AUROC = {roc_auc_score(y_te, err):.3f}")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    save(fig, "roc_curve")

    cm = confusion_matrix(y_te, err > thr)
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("navy", ["white", tone(NAVY, 0.85)]))
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"]); ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(length=0, top=False, right=False)
    save(fig, "confusion_matrix")

    with torch.no_grad():
        rec = model(torch.from_numpy(X_te)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0), sharey=True)
    for k, (ax, cls, name) in enumerate(zip(axes, [0, 1], ["Normal", "Anomaly"])):
        i = np.where(y_te == cls)[0][0]
        ax.fill_between(range(X_te.shape[1]), X_te[i], rec[i], color=tone(RED, 0.25), lw=0, label="Error")
        ax.plot(X_te[i], color=NAVY, lw=0.9, label="Input")
        ax.plot(rec[i], color=RED, lw=0.9, ls="--", label="Reconstruction")
        ax.set_xlabel(f"Time step\n({'ab'[k]}) {name}, MAE = {err[i]:.3f}")
    axes[0].set_ylabel("Scaled amplitude"); axes[0].legend(loc="lower right")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "reconstructions")


def train():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y = load_data()
    X_tr, (X_val, y_val), (X_te, y_te) = split(X, y)
    print(f"train normals {len(X_tr)} | val {len(X_val)} ({y_val.sum()} anom) | test {len(X_te)} ({y_te.sum()} anom)")

    lo, hi = X_tr.min(), X_tr.max()
    X_tr, X_val_s, X_te_s = scale(X_tr, lo, hi), scale(X_val, lo, hi), scale(X_te, lo, hi)

    model = AutoEncoder(X.shape[1])
    hist = fit(model, X_tr, X_val_s[y_val == 0])

    thr = pick_threshold(recon_error(model, X_val_s), y_val)
    err_te = recon_error(model, X_te_s)
    metrics = evaluate(err_te, y_te, thr)
    print(json.dumps(metrics, indent=2, default=float))

    # Save model, scaling stats, threshold, metrics, figures, example inputs
    torch.save({"state_dict": model.state_dict(), "lo": float(lo), "hi": float(hi), "threshold": thr},
               f"{OUT_DIR}/model.pt")
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    plot_all(hist, model, X_te_s, y_te, err_te, thr)
    idx = np.r_[np.where(y_te == 0)[0][:5], np.where(y_te == 1)[0][:5]]
    pd.DataFrame(X_te[idx]).to_csv(f"{OUT_DIR}/example_inputs.csv", header=False, index=False)
    print(f"saved to {OUT_DIR}/")


# Inference: CSV rows of 140 raw ECG values -> error + label
def predict(path):
    ckpt = torch.load(f"{OUT_DIR}/model.pt")
    X = pd.read_csv(path, header=None).values.astype(np.float32)
    model = AutoEncoder(X.shape[1])
    model.load_state_dict(ckpt["state_dict"])
    err = recon_error(model, scale(X, ckpt["lo"], ckpt["hi"]))
    for i, e in enumerate(err):
        print(f"sample {i}: error {e:.4f} -> {'ANOMALY' if e > ckpt['threshold'] else 'normal'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("--input", default=f"{OUT_DIR}/example_inputs.csv")
    a = p.parse_args()
    train() if a.mode == "train" else predict(a.input)
