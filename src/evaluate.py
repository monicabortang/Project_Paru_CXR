"""Langkah 4 - Evaluasi satu run pada semua set uji (+ data eksternal).

  python -m src.evaluate --run runs/S2_AP_matched/efficientnet_b0/seed0 \
      --splits /kaggle/working/splits --img_root /kaggle/working/data \
      [--external ext_kermany=/kaggle/working/external/kermany.csv]

Metrik pada ambang yang ditetapkan di VALIDASI SUMBER (spesifisitas 90%),
sehingga pergeseran domain juga tercermin pada sensitivitas/spesifisitas.
Keluaran (di folder run): preds_<test>.csv, metrics.json, cm_<test>.png
Baris ringkas juga ditambahkan ke results/all_results.csv.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .common import append_csv, binary_metrics, bootstrap_auc_ci, save_json  # noqa: E402
from .data import CXRDataset  # noqa: E402
from .models import build_model  # noqa: E402
from .train import predict  # noqa: E402


def load_run(run_dir: Path, device):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["model"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def measure_efficiency(model, img_size, device, n_iter=50):
    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    for bs in (1, 32):
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iter
        if bs == 1:
            res["latency_ms_bs1"] = dt * 1000
        else:
            res["throughput_img_s_bs32"] = bs / dt
    return res


def subgroup_metrics(df: pd.DataFrame, thr: float) -> dict:
    """S5: performa per jenis kelamin, kelompok usia, dan subkelas negatif."""
    out = {}
    df = df.copy()
    df["age_bin"] = pd.cut(df.age, [0, 40, 60, 120], right=False,
                           labels=["<40", "40-59", ">=60"])
    for col in ["sex", "age_bin", "view"]:
        if col not in df:
            continue
        for k, g in df.groupby(col, observed=True):
            if len(g) >= 30:
                out[f"{col}={k}"] = binary_metrics(g.label, g.prob, thr)
    # tingkat positif palsu per subkelas negatif
    if "class3" in df:
        for c in ["Normal", "No Lung Opacity / Not Normal"]:
            g = df[df.class3 == c]
            if len(g):
                out[f"FPR[{c}]"] = float((g.prob >= thr).mean())
    return out


def plot_cm(m: dict, title: str, path: Path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Negatif", "Positif"])
    ax.set_yticks([0, 1], ["Negatif", "Positif"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--tests", default="all,AP_natural,PA_natural,AP_matched,PA_matched")
    ap.add_argument("--external", action="append", default=[],
                    help="nama=path.csv (img_path absolut). Boleh diulang.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--results_csv", default="results/all_results.csv")
    ap.add_argument("--no_efficiency", action="store_true")
    a = ap.parse_args(argv)

    run_dir = Path(a.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(run_dir, device)
    thr = ck["thr_spec"]

    tests = {t: Path(a.splits) / "test" / f"{t}.csv" for t in a.tests.split(",") if t}
    for e in a.external:
        name, path = e.split("=", 1)
        tests[name] = Path(path)

    all_m = {"checkpoint": {k: ck[k] for k in
                            ("model", "scenario", "seed", "epoch", "val_auc", "thr_spec",
                             "thr_youden", "target_spec")}}
    if not a.no_efficiency:
        eff = measure_efficiency(model, ck["img_size"], device)
        eff["size_MB"] = os.path.getsize(run_dir / "best.pt") / 1e6
        eff["device"] = str(device)
        all_m["efficiency"] = eff

    for name, csv in tests.items():
        df = pd.read_csv(csv)
        ds = CXRDataset(df, a.img_root, ck["img_size"], train=False, clahe=ck["clahe"])
        dl = DataLoader(ds, a.batch_size, num_workers=a.workers, pin_memory=True)
        p, y, _ = predict(model, dl, device)
        df["prob"] = p
        keep = [c for c in ["patientId", "img_path", "label", "class3", "view", "age", "sex",
                            "boxes", "orig_h", "orig_w", "prob"] if c in df]
        df[keep].to_csv(run_dir / f"preds_{name}.csv", index=False)

        m = binary_metrics(y, p, thr)
        m["auc_ci95"] = bootstrap_auc_ci(y, p, a.n_boot)
        m["youden"] = binary_metrics(y, p, ck["thr_youden"])
        m["subgroups"] = subgroup_metrics(df, thr)
        all_m[name] = m
        plot_cm(m, f"{ck['scenario']} | {ck['model']} | uji {name}", run_dir / f"cm_{name}.png")
        print(f"{name:14s} n={m['n']:5d} prev={m['prevalence']:.3f} AUC={m['auc']:.4f} "
              f"[{m['auc_ci95'][0]:.3f},{m['auc_ci95'][1]:.3f}] macroF1={m['macro_f1']:.4f} "
              f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f}")
        append_csv({
            "scenario": ck["scenario"], "run_tag": run_dir.parent.parent.name,
            "model": ck["model"], "seed": ck["seed"], "test": name,
            "n": m["n"], "prevalence": m["prevalence"], "auc": m["auc"],
            "auc_lo": m["auc_ci95"][0], "auc_hi": m["auc_ci95"][1], "pr_auc": m["pr_auc"],
            "macro_f1": m["macro_f1"], "sensitivity": m["sensitivity"],
            "specificity": m["specificity"], "threshold": thr, "run_dir": str(run_dir),
        }, a.results_csv)

    save_json(all_m, run_dir / "metrics.json")
    print(f"Metrik tersimpan: {run_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
