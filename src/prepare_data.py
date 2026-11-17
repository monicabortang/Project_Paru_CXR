"""Langkah 1 - Konversi RSNA (DICOM) ke PNG + tabel metadata.

Keluaran:
  <out_dir>/images/<patientId>.png   (grayscale, sisi terpanjang = --size)
  <out_dir>/metadata.csv             satu baris per citra

Kolom metadata.csv:
  patientId, img_path, label (0/1), class3, view (AP/PA), age, sex,
  n_boxes, boxes (JSON [[x,y,w,h],...] koordinat asli), orig_h, orig_w,
  group_id (ID pasien untuk split; = NIH patient bila file pemetaan tersedia)

Contoh (Kaggle):
  python -m src.prepare_data \
      --rsna_dir /kaggle/input/rsna-pneumonia-detection-challenge \
      --out_dir /kaggle/working/data --size 512 --workers 4
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def read_labels(rsna_dir: Path) -> pd.DataFrame:
    lab = pd.read_csv(rsna_dir / "stage_2_train_labels.csv")
    cls = pd.read_csv(rsna_dir / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")

    def agg_boxes(g):
        g = g.dropna(subset=["x"])
        return json.dumps(g[["x", "y", "width", "height"]].round(1).values.tolist())

    try:  # pandas >= 2.2
        boxes = lab.groupby("patientId").apply(agg_boxes, include_groups=False)
    except TypeError:  # pandas lama (mis. bawaan Anaconda)
        boxes = lab.groupby("patientId").apply(agg_boxes)
    boxes = boxes.rename("boxes")
    target = lab.groupby("patientId")["Target"].max().rename("label")
    df = pd.concat([target, boxes], axis=1).reset_index()
    df = df.merge(cls, on="patientId", how="left").rename(columns={"class": "class3"})
    df["n_boxes"] = df["boxes"].map(lambda s: len(json.loads(s)))
    # Konsistensi: label positif <=> kelas 'Lung Opacity'
    bad = ((df.label == 1) != (df.class3 == "Lung Opacity")).sum()
    if bad:
        print(f"[PERINGATAN] {bad} baris label tidak konsisten dengan class3")
    return df


def _convert_one(args):
    """Baca 1 DICOM -> PNG. Dijalankan di proses terpisah."""
    dcm_path, png_path, size = args
    import cv2
    import pydicom

    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    img = arr.astype(np.uint8)
    h, w = img.shape
    if size and max(h, w) != size:
        scale = size / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if png_path is not None:
        cv2.imwrite(str(png_path), img)

    def _get(tag, default=None):
        v = getattr(ds, tag, default)
        return default if v in ("", None) else v

    age = _get("PatientAge")
    try:
        age = float(str(age).rstrip("Y"))
    except (TypeError, ValueError):
        age = np.nan
    return {
        "patientId": Path(dcm_path).stem,
        "view": str(_get("ViewPosition", "NA")).strip().upper(),
        "age": age,
        "sex": str(_get("PatientSex", "NA")).strip().upper(),
        "orig_h": h,
        "orig_w": w,
    }


def load_nih_mapping(path: Path) -> pd.DataFrame:
    """Pemetaan RSNA patientId -> ID pasien NIH (opsional).

    Menerima CSV/JSON yang punya satu kolom ID RSNA dan satu kolom nama file NIH
    (format 'PPPPPPPP_FFF.png'; 8 digit pertama = ID pasien NIH).
    Nama kolom dideteksi otomatis; sesuaikan bila file Anda berbeda.
    """
    if path.suffix == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_csv(path)
    rsna_col = next(c for c in df.columns if c.lower() in
                    ("patientid", "subset_img_id", "rsna_id", "img_id"))
    nih_col = next(c for c in df.columns if c.lower() in
                   ("orig_img_id", "nih_image", "image index", "image_index", "nih_img_id"))
    out = pd.DataFrame({
        "patientId": df[rsna_col].astype(str).str.replace(".dcm", "", regex=False),
        "group_id": "nih_" + df[nih_col].astype(str).str.split("_").str[0],
    })
    return out.drop_duplicates("patientId")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsna_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nih_map", default=None, help="opsional: file pemetaan RSNA->NIH")
    ap.add_argument("--limit", type=int, default=None, help="debug: hanya N citra")
    ap.add_argument("--meta_only", action="store_true", help="hanya baca header, tanpa PNG")
    a = ap.parse_args()

    rsna_dir, out_dir = Path(a.rsna_dir), Path(a.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Membaca label ...", flush=True)
    df = read_labels(rsna_dir)
    if a.limit:
        df = df.sample(a.limit, random_state=0)
    dcm_dir = rsna_dir / "stage_2_train_images"
    jobs = [
        (str(dcm_dir / f"{pid}.dcm"),
         None if a.meta_only else str(img_dir / f"{pid}.png"),
         a.size)
        for pid in df.patientId
    ]
    n = len(jobs)
    print(f"Mengonversi {n} citra dengan {a.workers} worker ... "
          "(progres ditampilkan tiap 500 citra; JANGAN tekan Ctrl+C)", flush=True)
    t0 = time.time()
    meta = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_convert_one, jobs, chunksize=32), 1):
            meta.append(r)
            if i % 500 == 0 or i == n:
                el = time.time() - t0
                eta = el / i * (n - i)
                print(f"  {i:6d}/{n} ({i / n:5.1%})  berjalan {el / 60:5.1f} mnt  "
                      f"sisa ~{eta / 60:5.1f} mnt", flush=True)
    meta = pd.DataFrame(meta)
    df = df.merge(meta, on="patientId", how="left")

    # Usia tidak masuk akal (RSNA memuat beberapa nilai > 100) -> NaN
    df.loc[(df.age < 0) | (df.age > 100), "age"] = np.nan
    df["img_path"] = "images/" + df.patientId + ".png"

    df["group_id"] = df.patientId  # default: tiap citra = 1 grup
    if a.nih_map:
        m = load_nih_mapping(Path(a.nih_map))
        df = df.drop(columns="group_id").merge(m, on="patientId", how="left")
        miss = df.group_id.isna().sum()
        df["group_id"] = df.group_id.fillna(df.patientId)
        print(f"Pemetaan NIH: {df.group_id.nunique()} pasien unik; {miss} citra tanpa pemetaan")

    cols = ["patientId", "img_path", "label", "class3", "view", "age", "sex",
            "n_boxes", "boxes", "orig_h", "orig_w", "group_id"]
    df[cols].to_csv(out_dir / "metadata.csv", index=False)
    print(df.groupby(["view", "class3"]).size().unstack(fill_value=0))
    print(f"Tersimpan: {out_dir/'metadata.csv'}")


if __name__ == "__main__":
    main()
