"""Langkah 2 - Pembagian data untuk semua skenario.

Prinsip:
1. Split di level GRUP (pasien), terstratifikasi menurut view x class3,
   sehingga satu pasien tidak pernah muncul di train dan test sekaligus.
2. Satu pool uji tetap (15%) dipakai SEMUA skenario -> antar skenario
   bisa dibandingkan pada citra uji yang identik.
3. Pengendalian faktor pengganggu AP/PA ("matched"):
   untuk setiap kelas (Lung Opacity / Not Normal / Normal) diambil jumlah
   citra yang SAMA dari AP dan PA. Hasilnya: ukuran data, prevalensi
   positif, dan komposisi kelas negatif identik antara AP dan PA.
   Versi "natural" (tanpa penyamaan) tetap dibuat sebagai pembanding.

Struktur keluaran (<out>/):
  test/all.csv, test/AP_natural.csv, test/PA_natural.csv,
  test/AP_matched.csv, test/PA_matched.csv
  S1_all/{train,val}.csv           latih semua posisi
  S2_AP_natural/{train,val}.csv    latih AP (apa adanya)
  S3_PA_natural/{train,val}.csv    latih PA (apa adanya)
  S2_AP_matched/{train,val}.csv    latih AP (disamakan dengan PA)
  S3_PA_matched/{train,val}.csv    latih PA (disamakan dengan AP)
  S4_AP2PA_f{1,5,10}/...           AP matched + 1/5/10% data PA
  S4_PA2AP_f{1,5,10}/...           PA matched + 1/5/10% data AP
  summary.csv                      statistik tiap file

Contoh:
  python -m src.make_splits --meta /kaggle/working/data/metadata.csv \
      --out /kaggle/working/splits --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import CLASS3


def group_split(df: pd.DataFrame, frac: float, seed: int):
    """Pisahkan df menjadi (sisa, bagian) dengan proporsi ~frac, per grup & terstratifikasi."""
    n_splits = max(2, round(1 / frac))
    strat = df["view"] + "|" + df["class3"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, part_idx = next(sgkf.split(df, strat, groups=df["group_id"]))
    return df.iloc[rest_idx].copy(), df.iloc[part_idx].copy()


def matched_pair(a: pd.DataFrame, b: pd.DataFrame, seed: int):
    """Ambil jumlah per-class3 yang sama dari a dan b (min dari keduanya)."""
    outs_a, outs_b = [], []
    for c in CLASS3:
        ca, cb = a[a.class3 == c], b[b.class3 == c]
        n = min(len(ca), len(cb))
        outs_a.append(ca.sample(n, random_state=seed))
        outs_b.append(cb.sample(n, random_state=seed))
    return (pd.concat(outs_a).sample(frac=1, random_state=seed),
            pd.concat(outs_b).sample(frac=1, random_state=seed))


def stratified_fraction(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Ambil frac dari df per grup pasien, terstratifikasi class3 (minimal 1 per kelas)."""
    parts = []
    for c in CLASS3:
        sub = df[df.class3 == c]
        groups = sub.group_id.drop_duplicates()
        k = max(1, int(round(frac * len(groups))))
        chosen = groups.sample(min(k, len(groups)), random_state=seed)
        parts.append(sub[sub.group_id.isin(chosen)])
    return pd.concat(parts)


def describe(name: str, df: pd.DataFrame) -> dict:
    row = {"file": name, "n": len(df), "n_pos": int(df.label.sum()),
           "prevalence": round(df.label.mean(), 4) if len(df) else np.nan,
           "n_AP": int((df.view == "AP").sum()), "n_PA": int((df.view == "PA").sum()),
           "n_groups": df.group_id.nunique(),
           "age_mean": round(df.age.mean(), 1),
           "male_pct": round((df.sex == "M").mean() * 100, 1)}
    for c in CLASS3:
        row[c] = int((df.class3 == c).sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--fracs", default="0.01,0.05,0.10")
    a = ap.parse_args()

    out = Path(a.out)
    df = pd.read_csv(a.meta)
    df = df[df.view.isin(["AP", "PA"])].reset_index(drop=True)
    print(f"Total citra AP/PA: {len(df)} | grup: {df.group_id.nunique()}")

    # --- 1. pool train / val / test (level grup) ------------------------
    trainval, test = group_split(df, a.test_frac, a.seed)
    train, val = group_split(trainval, a.val_frac / (1 - a.test_frac), a.seed + 1)
    assert not set(train.group_id) & set(test.group_id), "kebocoran grup train-test"
    assert not set(val.group_id) & set(test.group_id), "kebocoran grup val-test"
    assert not set(train.group_id) & set(val.group_id), "kebocoran grup train-val"

    files: dict[str, pd.DataFrame] = {}

    # --- 2. set uji (sama untuk semua skenario) -------------------------
    t_ap, t_pa = test[test.view == "AP"], test[test.view == "PA"]
    t_ap_m, t_pa_m = matched_pair(t_ap, t_pa, a.seed)
    files.update({
        "test/all": test, "test/AP_natural": t_ap, "test/PA_natural": t_pa,
        "test/AP_matched": t_ap_m, "test/PA_matched": t_pa_m,
    })

    # --- 3. skenario latih ----------------------------------------------
    tr_ap, tr_pa = train[train.view == "AP"], train[train.view == "PA"]
    va_ap, va_pa = val[val.view == "AP"], val[val.view == "PA"]
    tr_ap_m, tr_pa_m = matched_pair(tr_ap, tr_pa, a.seed)
    va_ap_m, va_pa_m = matched_pair(va_ap, va_pa, a.seed)

    files.update({
        "S1_all/train": train, "S1_all/val": val,
        "S2_AP_natural/train": tr_ap, "S2_AP_natural/val": va_ap,
        "S3_PA_natural/train": tr_pa, "S3_PA_natural/val": va_pa,
        "S2_AP_matched/train": tr_ap_m, "S2_AP_matched/val": va_ap_m,
        "S3_PA_matched/train": tr_pa_m, "S3_PA_matched/val": va_pa_m,
    })

    # --- 4. S4: efisiensi data target -----------------------------------
    for f in [float(x) for x in a.fracs.split(",")]:
        tag = f"f{round(f * 100)}"
        add_pa = stratified_fraction(tr_pa, f, a.seed)
        add_ap = stratified_fraction(tr_ap, f, a.seed)
        files[f"S4_AP2PA_{tag}/train"] = pd.concat([tr_ap_m, add_pa]).drop_duplicates("patientId")
        files[f"S4_AP2PA_{tag}/val"] = va_ap_m
        files[f"S4_PA2AP_{tag}/train"] = pd.concat([tr_pa_m, add_ap]).drop_duplicates("patientId")
        files[f"S4_PA2AP_{tag}/val"] = va_pa_m

    # --- 5. simpan --------------------------------------------------------
    rows = []
    for name, d in files.items():
        p = out / f"{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p, index=False)
        rows.append(describe(name, d))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.to_string(index=False))
    print(f"\nSplit tersimpan di {out} (seed={a.seed})")


if __name__ == "__main__":
    main()
