"""Pemuat model + inferensi satu citra (probabilitas + Grad-CAM) untuk API mobile."""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 2048  # citra lebih besar diperkecil dulu agar hemat memori


def _png_b64(img: np.ndarray) -> str:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def decode_image(data: bytes, filename: str = "") -> tuple[np.ndarray, dict]:
    """Bytes PNG/JPG/DICOM -> uint8 grayscale + info (mis. peringatan bukan X-ray)."""
    info: dict = {"warnings": []}
    is_dcm = filename.lower().endswith(".dcm") or data[128:132] == b"DICM"
    if is_dcm:
        import io

        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = np.percentile(arr, [0.5, 99.5])
        gray = (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        info["format"] = "DICOM"
        info["view"] = str(getattr(ds, "ViewPosition", "") or "")
    else:
        buf = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("File tidak dapat dibaca sebagai gambar (PNG/JPG/DICOM).")
        b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
        colorfulness = float((np.abs(r - g) + np.abs(g - b)).mean())
        if colorfulness > 12:
            info["warnings"].append(
                "Gambar tampak berwarna, kemungkinan bukan citra X-ray dada. Hasil tidak dapat diandalkan.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        info["format"] = "gambar"
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    if min(gray.shape) < 128:
        info["warnings"].append("Resolusi gambar sangat kecil; gunakan citra minimal 256 piksel.")
    ar = max(h, w) / max(min(h, w), 1)
    if ar > 1.6:
        info["warnings"].append("Rasio gambar tidak lazim untuk X-ray dada; pastikan gambar tidak terpotong.")
    return gray, info


class Predictor:
    """Membungkus checkpoint hasil src.train (best.pt).

    demo=True: memakai EfficientNet-B0 tanpa pelatihan, hanya untuk menguji
    alur aplikasi sebelum model selesai dilatih (hasilnya TIDAK bermakna).
    """

    def __init__(self, run_dir: str | None = None, demo: bool = False):
        import torch

        from src.evaluate import load_run
        from src.gradcam import GradCAM
        from src.models import build_model, gradcam_target

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if demo or not run_dir:
            model = build_model("efficientnet_b0", pretrained=False).to(self.device).eval()
            self.ck = {"model": "efficientnet_b0", "scenario": "DEMO (belum dilatih)",
                       "img_size": 224, "clahe": False, "thr_spec": 0.5, "val_auc": None,
                       "target_spec": None}
            self.demo = True
        else:
            model, self.ck = load_run(Path(run_dir), self.device)
            self.demo = False
        self.model = model
        layer, reshape = gradcam_target(model, self.ck["model"])
        self.gc = GradCAM(model, layer, reshape)
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"model": self.ck["model"], "scenario": self.ck["scenario"],
                "threshold": float(self.ck["thr_spec"]),
                "target_spec": self.ck.get("target_spec"),
                "val_auc": self.ck.get("val_auc"), "demo": self.demo,
                "device": str(self.device)}

    def predict(self, data: bytes, filename: str = "") -> dict:
        from src.data import pad_to_square, preprocess_array
        from src.gradcam import overlay

        t0 = time.time()
        gray, info = decode_image(data, filename)
        if self.ck.get("clahe"):
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        size = self.ck["img_size"]
        x = preprocess_array(gray, size).to(self.device)
        with self.lock:
            cam, prob = self.gc(x)
        sq = pad_to_square(gray)
        disp = 448
        ov = overlay(sq, cam, size)
        ov = cv2.resize(ov, (disp, disp), interpolation=cv2.INTER_LINEAR)
        orig = cv2.resize(sq, (disp, disp), interpolation=cv2.INTER_AREA)
        thr = float(self.ck["thr_spec"])
        positive = prob >= thr
        return {
            "probability": round(float(prob), 4),
            "threshold": round(thr, 4),
            "positive": bool(positive),
            "verdict": ("Terindikasi opasitas paru (curiga pneumonia)" if positive
                        else "Tidak terindikasi opasitas paru"),
            "gradcam": _png_b64(ov),
            "original": _png_b64(orig),
            "warnings": info["warnings"],
            "format": info.get("format"),
            "view": info.get("view", ""),
            "elapsed_ms": round((time.time() - t0) * 1000),
            **{k: v for k, v in self.info().items() if k in ("model", "scenario", "demo")},
        }
