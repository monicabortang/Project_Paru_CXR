"""Langkah 5 - Rekap semua hasil menjadi tabel laporan.

  python -m src.summarize --results results/all_results.csv --runs runs --out results/tables

Tabel yang dihasilkan:
  tabel_utama.csv        AUC/macro-F1/sens/spes (mean ± std antar seed) per skenario x uji
  tabel_domain_shift.csv penurunan lintas posisi, dua cara:
     drop = AUC(uji domain sendiri) - AUC(uji domain lain)    [model yang sama]
     gap  = AUC(model latih-target) - AUC(model latih-sumber) [set uji yang SAMA]
     'gap' adalah ukuran utama karena set ujinya identik, sehingga perbedaan
     tingkat kesulitan citra AP vs PA tidak ikut terhitung.
  tabel_uji_berpasangan.csv  selisih AUC + CI 95% bootstrap berpasangan
     (EfficientNet vs Swin; latih-AP vs latih-PA) memakai rata-rata prob antar seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import paired_bootstrap_auc_diff


def mean_std(s: pd.Series) -> str:
    return f"{s.mean():.4f} ± {s.std(ddof=0):.4f}" if len(s) > 1 else f"{s.mean():.4f}"


def ensemble_preds(runs: Path, run_tag: str, model: str, test: str):
    files = sorted((runs / run_tag / model).glob(f"seed*/preds_{test}.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f)[["patientId", "label", "prob"]] for f in files]
    base = dfs[0].rename(columns={"prob": "p0"})
    for i, d in enumerate(dfs[1:], 1):
        base = base.merge(d[["patientId", "prob"]].rename(columns={"prob": f"p{i}"}),
                          on="patientId")
    base["prob"] = base.filter(regex=r"^p\d+$").mean(axis=1)
    return base[["patientId", "label", "prob"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    out, runs = Path(a.out), Path(a.runs)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(a.results).drop_duplicates(["run_dir", "test"], keep="last")

    # 1. tabel utama
    t = (r.groupby(["run_tag", "model", "test"])
         .agg(n_seed=("seed", "nunique"), n=("n", "first"), prev=("prevalence", "first"),
              AUC=("auc", mean_std), macro_F1=("macro_f1", mean_std),
              sens=("sensitivity", mean_std), spec=("specificity", mean_std))
         .reset_index())
    t.to_csv(out / "tabel_utama.csv", index=False)

    # 2. domain shift (per seed -> mean ± std)
    rows = []
    for kind in ["matched", "natural"]:
        s2, s3 = f"S2_AP_{kind}", f"S3_PA_{kind}"
        t_ap, t_pa = f"AP_{kind}", f"PA_{kind}"
        for model in r.model.unique():
            def auc(tag, test):
                return r[(r.run_tag == tag) & (r.model == model) & (r.test == test)] \
                    .set_index("seed").auc
            a2_ap, a2_pa, a3_ap, a3_pa = auc(s2, t_ap), auc(s2, t_pa), auc(s3, t_ap), auc(s3, t_pa)
            if a2_ap.empty or a3_pa.empty:
                continue
            rows += [
                {"varian": kind, "model": model, "ukuran": "drop AP->PA (S2: uji AP - uji PA)",
                 "nilai": mean_std(a2_ap - a2_pa)},
                {"varian": kind, "model": model, "ukuran": "drop PA->AP (S3: uji PA - uji AP)",
                 "nilai": mean_std(a3_pa - a3_ap)},
                {"varian": kind, "model": model, "ukuran": "gap di uji PA (S3 - S2)",
                 "nilai": mean_std(a3_pa - a2_pa)},
                {"varian": kind, "model": model, "ukuran": "gap di uji AP (S2 - S3)",
                 "nilai": mean_std(a2_ap - a3_ap)},
            ]
    pd.DataFrame(rows).to_csv(out / "tabel_domain_shift.csv", index=False)

    # 3. uji berpasangan
    pairs = []
    models = sorted(r.model.unique())
    for (tag, test), _ in r.groupby(["run_tag", "test"]):
        if len(models) >= 2:
            pa = ensemble_preds(runs, tag, models[0], test)
            pb = ensemble_preds(runs, tag, models[1], test)
            if pa is not None and pb is not None:
                m = pa.merge(pb, on=["patientId", "label"], suffixes=("_a", "_b"))
                d, lo, hi, pv = paired_bootstrap_auc_diff(m.label, m.prob_a, m.prob_b, a.n_boot)
                pairs.append({"perbandingan": f"{models[0]} - {models[1]}", "skenario": tag,
                              "uji": test, "dAUC": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
    for kind in ["matched", "natural"]:
        for model in models:
            for test in [f"AP_{kind}", f"PA_{kind}"]:
                pa = ensemble_preds(runs, f"S2_AP_{kind}", model, test)
                pb = ensemble_preds(runs, f"S3_PA_{kind}", model, test)
                if pa is None or pb is None:
                    continue
                m = pa.merge(pb, on=["patientId", "label"], suffixes=("_a", "_b"))
                d, lo, hi, pv = paired_bootstrap_auc_diff(m.label, m.prob_a, m.prob_b, a.n_boot)
                pairs.append({"perbandingan": f"latih AP - latih PA ({model})",
                              "skenario": kind, "uji": test, "dAUC": d,
                              "ci_lo": lo, "ci_hi": hi, "p": pv})
    pd.DataFrame(pairs).round(4).to_csv(out / "tabel_uji_berpasangan.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_colwidth", 40):
        print(t.to_string(index=False))
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print(pd.DataFrame(pairs).round(4).to_string(index=False))
    print(f"\nTabel tersimpan di {out}")


if __name__ == "__main__":
    main()
