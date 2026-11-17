"""Langkah 3 - Pelatihan satu model pada satu skenario.

Contoh:
  python -m src.train --splits /kaggle/working/splits --scenario S1_all \
      --img_root /kaggle/working/data --model efficientnet_b0 --seed 0

Keluaran: runs/<scenario>/<model>/seed<k>/
  best.pt          bobot terbaik (AUC validasi tertinggi) + config + ambang
  history.csv      loss/AUC per epoch
  curves.png       kurva pelatihan
  val_preds.csv    prediksi validasi
  config.json      seluruh hyperparameter (untuk laporan)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .common import (save_json, seed_everything, threshold_at_specificity,  # noqa: E402
                     threshold_youden)
from .data import CXRDataset  # noqa: E402
from .models import MODEL_DEFAULTS, build_model, resolve_name  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--scenario", required=True, help="mis. S1_all, S2_AP_matched")
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--pos_weight", action="store_true",
                    help="bobot kelas positif = neg/pos pada BCE")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--target_spec", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=None, help="debug: batasi data latih")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="", help="akhiran nama run, mis. _clahe untuk ablasi")
    a = ap.parse_args(argv)
    a.model = resolve_name(a.model)
    d = MODEL_DEFAULTS.get(a.model, {})
    a.lr = a.lr or d.get("lr", 1e-4)
    a.batch_size = a.batch_size or d.get("batch_size", 32)
    a.img_size = a.img_size or d.get("img_size", 224)
    return a


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys, losses = [], [], []
    crit = torch.nn.BCEWithLogitsLoss()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logit = model(x).squeeze(1)
        losses.append(crit(logit.float(), y).item() * len(y))
        ps.append(torch.sigmoid(logit.float()).cpu())
        ys.append(y.cpu())
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    return p, y, float(np.sum(losses) / len(y))


def main(argv=None):
    a = get_args(argv)
    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(a.out) / f"{a.scenario}{a.tag}" / a.model / f"seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sdir = Path(a.splits) / a.scenario
    tr_df, va_df = pd.read_csv(sdir / "train.csv"), pd.read_csv(sdir / "val.csv")
    if a.max_train:
        tr_df = tr_df.sample(min(a.max_train, len(tr_df)), random_state=a.seed)
    ds_tr = CXRDataset(tr_df, a.img_root, a.img_size, train=True, clahe=a.clahe, hflip=a.hflip)
    ds_va = CXRDataset(va_df, a.img_root, a.img_size, train=False, clahe=a.clahe)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ds_tr, a.batch_size, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, generator=g,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, a.batch_size * 2, shuffle=False, num_workers=a.workers,
                       pin_memory=True, persistent_workers=a.workers > 0)

    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pos = tr_df.label.sum()
    pw = torch.tensor((len(tr_df) - pos) / max(pos, 1), device=device) if a.pos_weight else None
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * len(dl_tr)
    warm = int(a.warmup_epochs * len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    cfg = vars(a) | {"n_train": len(tr_df), "n_val": len(va_df),
                     "train_prevalence": float(tr_df.label.mean()),
                     "n_params": n_params, "device": str(device),
                     "torch": torch.__version__}
    try:
        import timm
        cfg["timm"] = timm.__version__
    except ImportError:
        pass
    save_json(cfg, run_dir / "config.json")
    print(f"[{a.scenario}|{a.model}|seed{a.seed}] train={len(tr_df)} val={len(va_df)} "
          f"params={n_params/1e6:.1f}M device={device}")

    hist, best_auc, bad = [], -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x).squeeze(1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        p, yv, vloss = predict(model, dl_va, device)
        vauc = roc_auc_score(yv, p) if len(np.unique(yv)) == 2 else float("nan")
        row = dict(epoch=ep, train_loss=run_loss / n, val_loss=vloss, val_auc=vauc,
                   lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
        hist.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()))
        if vauc > best_auc:
            best_auc, bad = vauc, 0
            thr_spec = threshold_at_specificity(yv, p, a.target_spec)
            thr_youden = threshold_youden(yv, p)
            torch.save({"model": a.model, "state_dict": model.state_dict(),
                        "img_size": a.img_size, "clahe": a.clahe,
                        "thr_spec": thr_spec, "thr_youden": thr_youden,
                        "target_spec": a.target_spec, "epoch": ep, "val_auc": vauc,
                        "scenario": a.scenario, "seed": a.seed}, run_dir / "best.pt")
            pd.DataFrame({"patientId": va_df.patientId, "label": yv, "prob": p}).to_csv(
                run_dir / "val_preds.csv", index=False)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping pada epoch {ep}")
                break

    h = pd.DataFrame(hist)
    h.to_csv(run_dir / "history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Loss BCE")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(h.epoch, h.val_auc, marker="o")
    ax[1].set_title("AUC validasi")
    ax[1].set_xlabel("epoch")
    fig.suptitle(f"{a.scenario} | {a.model} | seed {a.seed}")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=150)
    print(f"Selesai. AUC val terbaik={best_auc:.4f} -> {run_dir/'best.pt'}")
    return run_dir


if __name__ == "__main__":
    main()
