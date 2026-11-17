"""Langkah 6 - Grad-CAM dan kesesuaiannya dengan bounding box radiolog.

  python -m src.gradcam --run runs/S1_all/efficientnet_b0/seed0 \
      --test_csv runs/S1_all/efficientnet_b0/seed0/preds_all.csv \
      --img_root /kaggle/working/data --max_images 500

Metrik lokalisasi (dihitung pada citra POSITIF):
  pointing_game : 1 bila titik CAM tertinggi jatuh di dalam salah satu bbox
  energy_in_box : proporsi "energi" CAM yang berada di dalam bbox
  iou@0.5       : IoU antara CAM>=0.5*max dan gabungan bbox
  box_area_frac : luas bbox / luas citra (= nilai energy_in_box bila CAM acak)
  border_energy : energi CAM di bingkai luar 10% citra -> indikator shortcut
                  (penanda teks/"PORTABLE" biasanya di tepi citra)
Keluaran: gradcam_<nama>.csv, gradcam_summary.json, gradcam_examples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .common import load_gray, parse_boxes, save_json  # noqa: E402
from .data import pad_to_square, preprocess_array  # noqa: E402
from .evaluate import load_run  # noqa: E402
from .models import gradcam_target  # noqa: E402


class GradCAM:
    def __init__(self, model, layer, reshape=None):
        self.model, self.reshape = model, reshape
        self.act = None
        self.grad = None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = self.model(x).squeeze()
            logit.backward()
        a, g = self.act.detach(), self.grad.detach()
        if self.reshape:
            a, g = self.reshape(a), self.reshape(g)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, float(torch.sigmoid(logit).item())


def box_mask(boxes, orig_h, orig_w, size):
    """Bbox (koordinat asli) -> mask pada citra yang dipad persegi lalu di-resize."""
    s = max(orig_h, orig_w)
    oy, ox = (s - orig_h) / 2, (s - orig_w) / 2
    k = size / s
    m = np.zeros((size, size), dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = int((x + ox) * k), int((y + oy) * k)
        x1, y1 = int(np.ceil((x + w + ox) * k)), int(np.ceil((y + h + oy) * k))
        m[max(y0, 0):y1, max(x0, 0):x1] = True
    return m


def loc_metrics(cam, mask):
    iy, ix = np.unravel_index(np.argmax(cam), cam.shape)
    b = int(cam.shape[0] * 0.10)
    border = np.ones_like(cam, dtype=bool)
    border[b:-b, b:-b] = False
    bin_ = cam >= 0.5
    inter, union = (bin_ & mask).sum(), (bin_ | mask).sum()
    tot = cam.sum() + 1e-8
    return {"pointing_game": int(mask[iy, ix]),
            "energy_in_box": float((cam * mask).sum() / tot),
            "iou@0.5": float(inter / union) if union else 0.0,
            "box_area_frac": float(mask.mean()),
            "border_energy": float((cam * border).sum() / tot)}


def overlay(gray_sq, cam, size):
    img = cv2.resize(gray_sq, (size, size))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.6 * np.stack([img] * 3, -1) + 0.4 * heat).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test_csv", required=True, help="preds_<test>.csv dari evaluate")
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--n_examples", type=int, default=12)
    a = ap.parse_args()

    run_dir = Path(a.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(run_dir, device)
    layer, reshape = gradcam_target(model, ck["model"])
    gc = GradCAM(model, layer, reshape)
    size, thr = ck["img_size"], ck["thr_spec"]

    df = pd.read_csv(a.test_csv)
    pos = df[df.label == 1]
    pos = pos.sample(min(a.max_images, len(pos)), random_state=0)
    rows = []
    for _, r in pos.iterrows():
        p = Path(r.img_path)
        gray = load_gray(p if p.is_absolute() else Path(a.img_root) / p, ck["clahe"])
        x = preprocess_array(gray, size, False).to(device)  # CLAHE sudah diterapkan
        cam, prob = gc(x)
        mask = box_mask(parse_boxes(r.boxes), r.orig_h, r.orig_w, size)
        rows.append({"patientId": r.patientId, "view": r.get("view"), "prob": prob,
                     "tp": int(prob >= thr)} | loc_metrics(cam, mask))
    res = pd.DataFrame(rows)
    name = Path(a.test_csv).stem.replace("preds_", "")
    res.to_csv(run_dir / f"gradcam_{name}.csv", index=False)

    cols = ["pointing_game", "energy_in_box", "iou@0.5", "box_area_frac", "border_energy"]
    summ = {"n_positive": len(res),
            "all_positive": res[cols].mean().round(4).to_dict(),
            "true_positive": res[res.tp == 1][cols].mean().round(4).to_dict()}
    if "view" in res:
        summ["per_view"] = res.groupby("view")[cols].mean().round(4).to_dict("index")
    save_json(summ, run_dir / f"gradcam_summary_{name}.json")
    print(summ)

    # contoh visual: TP, FN, FP (FP tanpa bbox)
    picks = []
    tp = df[(df.label == 1) & (df.prob >= thr)].nlargest(a.n_examples // 3, "prob")
    fn = df[(df.label == 1) & (df.prob < thr)].nsmallest(a.n_examples // 3, "prob")
    fp = df[(df.label == 0) & (df.prob >= thr)].nlargest(a.n_examples // 3, "prob")
    for tag, sub in [("TP", tp), ("FN", fn), ("FP", fp)]:
        picks += [(tag, r) for _, r in sub.iterrows()]
    if picks:
        ncol = 4
        nrow = int(np.ceil(len(picks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (tag, r) in zip(np.atleast_1d(axes).ravel(), picks):
            p = Path(r.img_path)
            gray = load_gray(p if p.is_absolute() else Path(a.img_root) / p, ck["clahe"])
            x = preprocess_array(gray, size, False).to(device)
            cam, prob = gc(x)
            ax.imshow(overlay(pad_to_square(gray), cam, size))
            m = box_mask(parse_boxes(r.boxes), r.orig_h, r.orig_w, size)
            if m.any():
                ax.contour(m, levels=[0.5], colors="lime", linewidths=1)
            ax.set_title(f"{tag} {r.get('view', '')} p={prob:.2f}", fontsize=8)
        fig.suptitle(f"Grad-CAM {ck['model']} ({ck['scenario']}) - kotak hijau: bbox radiolog",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(run_dir / f"gradcam_examples_{name}.png", dpi=150)
    print(f"Grad-CAM tersimpan di {run_dir}")


if __name__ == "__main__":
    main()
