"""Demo Gradio (Progress 7): unggah X-ray dada -> probabilitas + Grad-CAM.

  python app.py --run runs/S1_all/swin_tiny_patch4_window7_224/seed0 [--share]

Bisa dipasang di Hugging Face Spaces (salin app.py, src/, requirements.txt, best.pt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from src.data import pad_to_square, preprocess_array
from src.evaluate import load_run
from src.gradcam import GradCAM, overlay
from src.models import gradcam_target

DISCLAIMER = ("Prototipe penelitian mahasiswa - BUKAN alat diagnosis. "
              "Hasil harus dikonfirmasi oleh dokter/radiolog.")


def read_any(file_path: str) -> np.ndarray:
    import cv2

    if file_path.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(file_path)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = np.percentile(arr, [0.5, 99.5])
        return (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def build_app(run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run(run_dir, device)
    layer, reshape = gradcam_target(model, ck["model"])
    gc = GradCAM(model, layer, reshape)
    thr = ck["thr_spec"]

    def infer(file):
        if file is None:
            return None, "Unggah citra terlebih dahulu."
        gray = read_any(file if isinstance(file, str) else file.name)
        if ck["clahe"]:
            import cv2
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        x = preprocess_array(gray, ck["img_size"]).to(device)
        cam, prob = gc(x)
        verdict = "TERINDIKASI opasitas pneumonia" if prob >= thr else "Tidak terindikasi"
        txt = (f"**Probabilitas: {prob:.3f}** (ambang {thr:.3f}) -> **{verdict}**\n\n"
               f"Model: `{ck['model']}` | skenario latih: `{ck['scenario']}`\n\n_{DISCLAIMER}_")
        return overlay(pad_to_square(gray), cam, ck["img_size"]), txt

    with gr.Blocks(title="Deteksi Pneumonia CXR") as demo:
        gr.Markdown("# Deteksi Dini Pneumonia dari X-ray Dada\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.File(label="Citra X-ray (PNG/JPG/DICOM)",
                          file_types=[".png", ".jpg", ".jpeg", ".dcm"])
            out_img = gr.Image(label="Grad-CAM (area yang memengaruhi prediksi)")
        out_txt = gr.Markdown()
        gr.Button("Analisis").click(infer, inp, [out_img, out_txt])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_app(Path(a.run)).launch(share=a.share)
