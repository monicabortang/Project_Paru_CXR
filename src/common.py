"""Fungsi bersama: seed, pemuatan citra, metrik, dan utilitas I/O.

Modul ini sengaja tidak mengimpor torch di level atas agar bisa dipakai
oleh skrip persiapan data (prepare_data, make_splits) tanpa GPU/torch.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

CLASS3 = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"]
VIEWS = ["AP", "PA"]


# ---------------------------------------------------------------- seed
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # True memperlambat; cukup seed
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


# ---------------------------------------------------------------- citra
def load_gray(path: str | Path, clahe: bool = False):
    """Membaca citra sebagai uint8 grayscale (H, W)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img


def parse_boxes(s) -> list[list[float]]:
    """Kolom 'boxes' disimpan sebagai JSON [[x, y, w, h], ...] (koordinat asli)."""
    if isinstance(s, str) and s:
        return json.loads(s)
    return []


# ---------------------------------------------------------------- metrik
def threshold_at_specificity(y, p, target_spec: float = 0.90) -> float:
    """Ambang terkecil yang menghasilkan spesifisitas >= target pada data validasi."""
    y = np.asarray(y)
    p = np.asarray(p)
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return 0.5
    k = int(np.ceil(target_spec * len(neg))) - 1
    k = min(max(k, 0), len(neg) - 1)
    return float(np.nextafter(neg[k], np.inf))


def threshold_youden(y, p) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(min(thr[j], 1.0))


def binary_metrics(y, p, thr: float) -> dict:
    """Metrik pada ambang tertentu + metrik bebas ambang (AUC, AP)."""
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(thr),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    else:
        out["auc"] = out["pr_auc"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def bootstrap_auc_ci(y, p, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    aucs = []
    for _ in range(n_boot):  # bootstrap terstratifikasi agar prevalensi tetap
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 0):
    """Selisih AUC (A - B) pada citra uji yang SAMA + CI 95% dan p-value dua sisi."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    base = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(base), float(lo), float(hi), float(min(pval, 1.0))


# ---------------------------------------------------------------- I/O
def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def append_csv(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
