"""Membuat kaggle_bundle.zip agar notebook Kaggle bisa berjalan TANPA Internet.

Isi bundle:
  repo/     kode (src, scripts, splits, requirements)  -> pengganti git clone
  hf_hub/   bobot pretrained timm (EfficientNet-B0, Swin-T) -> pengganti unduhan HF
  wheels/   paket pip (timm, pydicom, huggingface_hub, safetensors) -> pengganti pip install

Jalankan di laptop (yang ada Internet), dari folder project:
  python scripts/make_kaggle_bundle.py
Lalu unggah kaggle_bundle.zip ke Kaggle: panel kanan notebook -> Upload -> New Dataset.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_bundle"
MODELS = ["efficientnet_b0", "swin_tiny_patch4_window7_224"]
WHEELS = ["timm", "pydicom", "huggingface_hub", "safetensors"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "repo").mkdir(parents=True)

    # 1. kode + split
    for name in ["src", "scripts", "splits"]:
        shutil.copytree(ROOT / name, OUT / "repo" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ["requirements.txt", "README.md"]:
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, OUT / "repo" / f)
    print("[1/3] kode + split disalin")

    # 2. bobot pretrained -> cache Hugging Face lokal, lalu salin
    import timm
    from huggingface_hub import constants

    for m in MODELS:
        print(f"      memastikan bobot {m} tersedia ...", flush=True)
        timm.create_model(m, pretrained=True)
    cache = Path(constants.HF_HUB_CACHE)
    n = 0
    for d in cache.glob("models--timm--*"):
        if any(m in d.name for m in MODELS):
            shutil.copytree(d, OUT / "hf_hub" / d.name, symlinks=False)
            n += 1
    if n == 0:
        sys.exit(f"Bobot tidak ditemukan di {cache}")
    print(f"[2/3] {n} folder bobot disalin dari {cache}")

    # 3. wheel pip (untuk berjaga bila paket tidak ada di Kaggle)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                    "--python-version", "3.13", "--platform", "manylinux2014_x86_64",
                    "-d", str(OUT / "wheels"), *WHEELS], check=False)
    print("[3/3] wheel diunduh (bila gagal, notebook memakai paket bawaan Kaggle)")

    zpath = ROOT / "kaggle_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
    shutil.rmtree(OUT)
    print(f"\nSelesai: {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    print("Unggah file ini di Kaggle: panel kanan notebook -> Upload -> New Dataset.")


if __name__ == "__main__":
    main()
