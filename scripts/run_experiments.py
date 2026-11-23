"""Menjalankan grid eksperimen (latih + evaluasi), dapat dilanjutkan (resume).

Run yang sudah selesai (history.csv + metrics.json) akan dilewati, sehingga aman
dijalankan ulang di sesi Kaggle berikutnya (simpan folder runs/ sebagai output).

  python scripts/run_experiments.py --group core \
      --splits /kaggle/working/splits --img_root /kaggle/working/data \
      [--external kermany=/kaggle/working/external/kermany/kermany.csv]

Grup:
  baseline : S1_all x efficientnet_b0 x seed 0,1,2                         (3 run, Progress 3)
  core     : S1_all, S2_AP_matched, S3_PA_matched  x 2 model x seed 0,1,2  (18 run)
  natural  : S2_AP_natural, S3_PA_natural          x 2 model x seed 0      (4 run)
  s4       : S4_*_f1/f5/f10                        x 2 model x seed 0      (12 run)
  ablation : S1_all + CLAHE ; S1_all + pos_weight  x 2 model x seed 0      (4 run)
  smoke    : uji cepat 1 epoch, 256 citra (cek pipeline sebelum run panjang)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = ["efficientnet_b0", "swin_tiny_patch4_window7_224"]

GROUPS = {
    "baseline": [("S1_all", "efficientnet_b0", sd, []) for sd in (0, 1, 2)],  # Progress 3
    "core": [(s, m, sd, []) for s in ["S1_all", "S2_AP_matched", "S3_PA_matched"]
             for m in MODELS for sd in (0, 1, 2)],
    "natural": [(s, m, 0, []) for s in ["S2_AP_natural", "S3_PA_natural"] for m in MODELS],
    "s4": [(f"S4_{d}_f{f}", m, 0, []) for d in ["AP2PA", "PA2AP"] for f in (1, 5, 10)
           for m in MODELS],
    "ablation": [("S1_all", m, 0, ["--clahe", "--tag", "_clahe"]) for m in MODELS]
                + [("S1_all", m, 0, ["--pos_weight", "--tag", "_posw"]) for m in MODELS],
    "smoke": [("S1_all", m, 0, ["--epochs", "1", "--max_train", "256", "--tag", "_smoke"])
              for m in MODELS],
}


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--splits", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--external", action="append", default=[])
    ap.add_argument("--workers", default="4")
    ap.add_argument("--models", default=None, help="batasi model, pisahkan koma")
    a = ap.parse_args()

    only = set(a.models.split(",")) if a.models else None
    for scen, model, seed, extra in GROUPS[a.group]:
        if only and model not in only:
            continue
        tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
        rd = Path(a.runs) / f"{scen}{tag}" / model / f"seed{seed}"
        # history.csv baru ditulis setelah pelatihan selesai -> run yang terputus diulang
        if not (rd / "history.csv").exists():
            run([sys.executable, "-m", "src.train", "--splits", a.splits, "--scenario", scen,
                 "--img_root", a.img_root, "--model", model, "--seed", str(seed),
                 "--out", a.runs, "--workers", a.workers, *extra])
        if not (rd / "metrics.json").exists():
            ev = [sys.executable, "-m", "src.evaluate", "--run", str(rd), "--splits", a.splits,
                  "--img_root", a.img_root, "--workers", a.workers]
            for e in a.external:
                ev += ["--external", e]
            if a.group == "smoke":
                ev += ["--n_boot", "50", "--results_csv", "results/smoke_results.csv"]
            run(ev)
        else:
            print(f"[lewati] {rd} sudah selesai")


if __name__ == "__main__":
    main()
