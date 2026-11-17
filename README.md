# Project_Paru_CXR — Deteksi Pneumonia pada X-ray Dada & Ketahanan terhadap Pergeseran Posisi AP/PA

Project Mata Kuliah Kecerdasan Buatan — Topik **RCH-2: AI untuk Deteksi Dini Penyakit dari Citra Medis**
Kelompok 08 · Dataset utama: RSNA Pneumonia Detection Challenge · Uji eksternal: Kermany (Guangzhou)

## Pertanyaan penelitian
1. Seberapa baik CNN (EfficientNet-B0) dan transformer (Swin-T) mendeteksi opasitas pneumonia pada RSNA (in-domain)?
2. Seberapa besar penurunan performa bila model dilatih pada satu posisi (AP) lalu diuji pada posisi lain (PA), dan sebaliknya, **setelah faktor pengganggu disamakan**?
3. Apakah area Grad-CAM sesuai dengan bounding box radiolog?
4. Seberapa jauh model bertahan di rumah sakit/perangkat lain (uji eksternal)?

## Model
| Peran | Model (timm) | Parameter | Alasan |
|---|---|---|---|
| Baseline | `efficientnet_b0` | 5,3 jt | CNN ringan dan kuat; cepat di GPU Kaggle |
| Model utama | `swin_tiny_patch4_window7_224` | 28 jt | Transformer hierarkis; lebih efisien data daripada ViT-B/16 pada data ±26 ribu citra; mudah dibuat Grad-CAM |

Keduanya memakai bobot ImageNet, input 224×224, AdamW + cosine LR + warmup 1 epoch, AMP, early stopping (AUC validasi, patience 4).

## Rancangan pembagian data & pengendalian faktor pengganggu
- **Split per grup pasien**, terstratifikasi *posisi × kelas* (StratifiedGroupKFold): 70% latih / 15% validasi / 15% uji.
  `patientId` RSNA pada dasarnya ID per citra. Bila file pemetaan RSNA→NIH tersedia (`--nih_map`), grup = ID pasien NIH sehingga pasien yang sama tidak mungkin muncul di latih dan uji sekaligus. Tanpa file itu, keterbatasan ini dicantumkan di laporan.
- **Satu pool uji tetap** dipakai semua skenario → perbandingan antarskenario memakai citra uji yang sama.
- **Penyamaan (matched) AP/PA:** untuk setiap kelas (*Lung Opacity*, *No Lung Opacity/Not Normal*, *Normal*) diambil jumlah citra yang sama dari AP dan PA. Ukuran data, prevalensi, dan komposisi kelas negatif menjadi identik, sehingga selisih performa tidak bisa dijelaskan oleh perbedaan prevalensi atau jumlah data. Versi *natural* tetap dilatih sebagai pembanding.
- **Dua ukuran pergeseran domain** (`src/summarize.py`):
  - *drop* = AUC(uji domain sendiri) − AUC(uji domain lain) untuk model yang sama.
  - *gap* = AUC(model latih-target) − AUC(model latih-sumber) **pada set uji yang sama** → ukuran utama, karena tidak tercampur perbedaan tingkat kesulitan citra AP vs PA.
- **AUC** = metrik utama (tidak bergantung prevalensi). Ambang klasifikasi ditetapkan di validasi sumber pada spesifisitas 90%, lalu dipakai apa adanya di domain lain.
- Signifikansi: CI 95% bootstrap terstratifikasi; selisih AUC antarmodel/antarskenario dengan bootstrap berpasangan.

## Skenario
| Kode | Latih | Uji | Tujuan |
|---|---|---|---|
| S1 | Semua posisi | all, AP, PA | Acuan in-domain |
| S2 | AP (matched & natural) | AP & PA | Pergeseran AP→PA |
| S3 | PA (matched & natural) | AP & PA | Pergeseran PA→AP |
| S4 | Sumber matched + 1/5/10% target | target | Efisiensi data target |
| S5 | (model S1) | subkelompok usia, jenis kelamin, subkelas negatif | Kesenjangan performa |
| S6 | (semua model) | Kermany / data lokal | Generalisasi antar rumah sakit & perangkat |
| Ablasi | S1 + CLAHE; S1 + pos_weight | sama | Pengaruh prapemrosesan & bobot kelas |

## Struktur repositori
```
src/prepare_data.py   DICOM -> PNG + metadata (view, usia, jenis kelamin, bbox)
src/eda.py            tabel & grafik EDA (Progress 2)
src/make_splits.py    split per pasien, set uji tetap, matched/natural, S4
src/external.py       data uji eksternal (Kermany / CSV data lokal)
src/data.py           Dataset + augmentasi
src/models.py         model timm + layer target Grad-CAM
src/train.py          pelatihan satu run
src/evaluate.py       metrik, CI, subkelompok, confusion matrix, efisiensi
src/gradcam.py        Grad-CAM + pointing game / energy-in-box / IoU / border energy
src/summarize.py      tabel laporan (mean ± std antar seed, domain shift, uji berpasangan)
scripts/run_experiments.py  grid eksperimen yang bisa dilanjutkan
notebooks/kaggle_runner.ipynb  urutan lengkap di Kaggle
app.py                demo Gradio (Progress 7)
```

## Cara menjalankan (Kaggle)
Buka `notebooks/kaggle_runner.ipynb` di Kaggle (GPU T4, Internet On, tambahkan data kompetisi RSNA + `paultimothymooney/chest-xray-pneumonia`). Secara manual:
```bash
python -m src.prepare_data --rsna_dir /kaggle/input/rsna-pneumonia-detection-challenge --out_dir /kaggle/working/data
python -m src.eda          --meta /kaggle/working/data/metadata.csv --img_root /kaggle/working/data
python -m src.make_splits  --meta /kaggle/working/data/metadata.csv --out /kaggle/working/splits --seed 42
python -m src.external kermany --src /kaggle/input/chest-xray-pneumonia/chest_xray --out /kaggle/working/external/kermany
python scripts/run_experiments.py --group smoke --splits /kaggle/working/splits --img_root /kaggle/working/data
python scripts/run_experiments.py --group core  --splits /kaggle/working/splits --img_root /kaggle/working/data \
       --external kermany=/kaggle/working/external/kermany/kermany.csv
python -m src.summarize
python app.py --run runs/S1_all/swin_tiny_patch4_window7_224/seed0 --share
```

## Estimasi komputasi (GPU T4, perkiraan kasar)
| Grup | Run | Perkiraan |
|---|---|---|
| prepare_data | 1× | 15–25 menit |
| core | 18 | 6–9 jam |
| natural + s4 + ablasi | 20 | 4–6 jam |
Kuota GPU Kaggle ±30 jam/minggu → cukup, asalkan dibagi ke beberapa sesi (runner otomatis melanjutkan).

## Reprodusibilitas
Seed split = 42; seed pelatihan = 0, 1, 2. Semua hyperparameter tersimpan di `runs/**/config.json`, metrik di `metrics.json`, dan rekap di `results/all_results.csv`.

## Penggunaan AI generatif
Kerangka kode disusun dengan bantuan Claude (Anthropic) lalu ditinjau, dijalankan, dan diuji oleh anggota kelompok.

> Prototipe penelitian — bukan alat diagnosis klinis.

## Aplikasi mobile ParuScan (PWA)
Web app untuk HP: ambil foto/pilih file X-ray (PNG/JPG/DICOM), lalu tampil probabilitas, keputusan pada ambang validasi, dan peta Grad-CAM. Gambar tidak disimpan di server; riwayat hanya disimpan di HP.

```bash
pip install -r requirements-mobile.txt
# sebelum model dilatih (uji tampilan saja):
python -m mobile_app.server --demo
# setelah pelatihan:
python -m mobile_app.server --run runs/S1_all/swin_tiny_patch4_window7_224/seed0
```
Buka alamat `http://<IP-laptop>:8000` yang dicetak server dari browser HP (satu Wi-Fi dengan laptop). Untuk akses dari mana saja via HTTPS (dan instal "Add to Home Screen"), deploy ke Hugging Face Spaces memakai `mobile_app/Dockerfile`.
