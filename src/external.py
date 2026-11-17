"""S6 - Siapkan data uji EKSTERNAL (rumah sakit/perangkat lain).

Mode 1 - Kermany "Chest X-Ray Images (Pneumonia)" (Guangzhou, pasien anak):
  Kaggle: paultimothymooney/chest-xray-pneumonia
  python -m src.external kermany \
      --src /kaggle/input/chest-xray-pneumonia/chest_xray \
      --out /kaggle/working/external/kermany --subsets test

Mode 2 - CSV generik (mis. data lokal rumah sakit/puskesmas yang sudah anonim):
  CSV wajib punya kolom: path (absolut / relatif ke --src), label (0/1)
  python -m src.external csv --src /path/folder --csv daftar.csv \
      --out /kaggle/working/external/lokal

Keluaran: <out>/images/*.png dan <out>/<nama>.csv (img_path absolut) yang
dapat langsung dipakai:  python -m src.evaluate ... --external kermany=<out>/kermany.csv

CATATAN METODOLOGIS (wajib dibahas di laporan):
- Label Kermany = diagnosis pneumonia (bakteri/virus), sedangkan label RSNA =
  "opasitas paru yang konsisten dengan pneumonia". Definisinya tidak identik.
- Kermany berisi pasien anak 1-5 tahun; RSNA mayoritas dewasa. Jadi S6 menguji
  pergeseran rumah sakit + perangkat + populasi sekaligus (bukan murni rumah sakit).
- Ambang klasifikasi tetap dari validasi RSNA; AUC adalah metrik utama S6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def to_png(src: Path, dst: Path, size: int):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    s = size / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return img.shape


def build_rows(items, out: Path, size: int, prefix: str):
    rows = []
    for i, (src, label) in enumerate(items):
        dst = out / "images" / f"{prefix}_{i:05d}.png"
        shape = to_png(src, dst, size)
        if shape is None:
            print(f"[lewati] gagal membaca {src}")
            continue
        rows.append({"patientId": f"{prefix}_{i:05d}", "img_path": str(dst.resolve()),
                     "label": int(label),
                     "class3": "Lung Opacity" if label else "Normal",
                     "view": "EXT", "age": float("nan"), "sex": "NA",
                     "n_boxes": 0, "boxes": "[]", "orig_h": shape[0], "orig_w": shape[1],
                     "group_id": f"{prefix}_{i:05d}", "source_file": str(src)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kermany", "csv"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--subsets", default="test", help="kermany: test / train,val,test")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    items = []
    if a.mode == "kermany":
        for sub in a.subsets.split(","):
            for cls, lab in [("NORMAL", 0), ("PNEUMONIA", 1)]:
                for f in sorted((src / sub / cls).glob("*.jp*g")):
                    items.append((f, lab))
        name = a.name or "kermany"
    else:
        d = pd.read_csv(a.csv)
        for p, lab in zip(d["path"], d["label"]):
            p = Path(p)
            items.append((p if p.is_absolute() else src / p, lab))
        name = a.name or "lokal"

    df = build_rows(items, out, a.size, name)
    df.to_csv(out / f"{name}.csv", index=False)
    print(df.label.value_counts().rename({0: "negatif", 1: "positif"}))
    print(f"Tersimpan: {out/(name + '.csv')}  (n={len(df)})")


if __name__ == "__main__":
    main()
