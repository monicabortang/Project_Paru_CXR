"""Eksplorasi data (luaran Progress 2): tabel statistik + grafik + contoh citra.

  python -m src.eda --meta /kaggle/working/data/metadata.csv \
      --img_root /kaggle/working/data --out results/eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import CLASS3, load_gray, parse_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--img_root", default=None)
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.meta)

    # Tabel 1: view x class3
    t1 = pd.crosstab(df.view, df.class3, margins=True, margins_name="Total")
    t1["Prevalensi positif (%)"] = (t1["Lung Opacity"] / t1["Total"] * 100).round(1)
    t1.to_csv(out / "tabel_view_x_kelas.csv")
    print(t1, "\n")

    # Tabel 2: demografi per view
    t2 = df.groupby("view").agg(
        n=("patientId", "size"),
        usia_mean=("age", "mean"), usia_std=("age", "std"),
        usia_missing=("age", lambda s: s.isna().sum()),
        pria_pct=("sex", lambda s: (s == "M").mean() * 100),
        bbox_per_positif=("n_boxes", lambda s: s[s > 0].mean()),
    ).round(2)
    t2.to_csv(out / "tabel_demografi_per_view.csv")
    print(t2, "\n")

    # Grafik 1: distribusi kelas per view
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(df.view, df.class3)[CLASS3].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Jumlah citra")
    ax.set_title("Distribusi kelas per posisi pengambilan")
    fig.tight_layout()
    fig.savefig(out / "grafik_kelas_per_view.png", dpi=150)

    # Grafik 2: usia per view
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in ["AP", "PA"]:
        ax.hist(df[df.view == v].age.dropna(), bins=30, alpha=0.6, label=v, density=True)
    ax.set_xlabel("Usia (tahun)")
    ax.set_ylabel("Densitas")
    ax.legend()
    ax.set_title("Distribusi usia per posisi pengambilan")
    fig.tight_layout()
    fig.savefig(out / "grafik_usia_per_view.png", dpi=150)

    # Grafik 3: contoh citra per kelas x view (dengan bbox)
    if a.img_root:
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for i, v in enumerate(["AP", "PA"]):
            for j, c in enumerate(CLASS3):
                sub = df[(df.view == v) & (df.class3 == c)]
                axx = axes[i, j]
                axx.axis("off")
                if not len(sub):
                    continue
                r = sub.iloc[rng.integers(len(sub))]
                try:
                    img = load_gray(Path(a.img_root) / r.img_path)
                except FileNotFoundError:
                    print(f"[lewati] citra tidak ditemukan: {r.img_path}")
                    continue
                s = img.shape[0] / r.orig_h
                axx.imshow(img, cmap="gray")
                for x, y, w, h in parse_boxes(r.boxes):
                    axx.add_patch(patches.Rectangle((x * s, y * s), w * s, h * s,
                                                    fill=False, ec="red", lw=1.5))
                axx.set_title(f"{v} - {c}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "contoh_citra.png", dpi=150)
    print(f"EDA tersimpan di {out}")


if __name__ == "__main__":
    main()
