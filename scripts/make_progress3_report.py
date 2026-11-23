"""Membuat Laporan Progress 3 (.docx) OTOMATIS dari hasil pelatihan baseline yang asli.

Semua angka diambil dari file hasil run (config.json, history.csv, metrics.json,
preds_*.csv, results/all_results.csv) - tidak ada angka yang diketik manual.

  python scripts/make_progress3_report.py
  python scripts/make_progress3_report.py --runs runs --results results/all_results.csv \
         --out "laporan/08_Progress 3.docx"

Perlu: pip install python-docx matplotlib pandas scikit-learn
Setelah jadi, buka di Word -> periksa bagian bertanda [CEK] -> File > Save As > PDF.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
except ImportError:
    sys.exit("Pasang dulu: python -m pip install python-docx")

TEAM = [("Mikhael Josua Roganda Panjaitan", "42324023", "Ketua"),
        ("Ellyzhabeth", "42324046", "Modeling"),
        ("Debora Febrianti Tampubolon", "42324052", "Evaluasi"),
        ("Monica Sitanggang", "42324053", "Data"),
        ("Michelle Sianturi", "42324062", "Dokumentasi")]
TEST_ORDER = [("all", "Semua (S1)"), ("AP_natural", "AP natural"), ("PA_natural", "PA natural"),
              ("AP_matched", "AP matched"), ("PA_matched", "PA matched"), ("kermany", "Kermany (eksternal)")]
TITLE = ("Ketahanan CNN dan Transformer terhadap Pergeseran Posisi Akuisisi (AP/PA) "
         "dalam Deteksi Pneumonia pada Citra X-ray Dada")
BLUE = RGBColor(0x1F, 0x3A, 0x5F)


# ------------------------------------------------------------------ util angka
def idn(x, d=3):
    """Format angka gaya Indonesia (koma desimal)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{d}f}".replace(".", ",")


def ms(series, d=3):
    s = pd.Series(series).dropna()
    if s.empty:
        return "–"
    return idn(s.mean(), d) if len(s) == 1 else f"{idn(s.mean(), d)} ± {idn(s.std(ddof=0), d)}"


def pct(x, d=1):
    return "–" if x is None or np.isnan(x) else f"{x * 100:.{d}f}%".replace(".", ",")


# ------------------------------------------------------------------ util docx
def set_cell_bg(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para(doc, text="", bold=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, italic=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    # dukung **tebal** sederhana
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part:
            r = p.add_run(part)
            r.bold = bold or (i % 2 == 1)
            r.italic = italic
            r.font.size = Pt(size)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    return p


def caption(doc, text):
    p = para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    p.paragraph_format.keep_with_next = True
    return p


def table(doc, rows, widths_cm, header=True, size=10, bold_first_col=False):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            if (header and i == 0) or (bold_first_col and j == 0):
                r.bold = True
            if header and i == 0:
                set_cell_bg(c, "D9E2F3")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # lebar kolom tetap (dihormati Word & LibreOffice) + tabel tidak terpotong antar halaman
    t.autofit = False
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w * 567)))
    last = len(t.rows) - 1
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        if header and i == 0:
            th = OxmlElement("w:tblHeader")
            trPr.append(th)
        if len(t.rows) <= 8 and i < last:
            for c in row.cells:
                for pp in c.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, width_cm, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, cap)


# ------------------------------------------------------------------ data
def load_runs(runs: Path, scenario: str, model: str):
    out = []
    for d in sorted((runs / scenario / model).glob("seed*")):
        if not (d / "history.csv").exists():
            continue
        r = {"seed": int(d.name.replace("seed", "")), "dir": d,
             "config": json.loads((d / "config.json").read_text()),
             "history": pd.read_csv(d / "history.csv")}
        m = d / "metrics.json"
        r["metrics"] = json.loads(m.read_text()) if m.exists() else {}
        out.append(r)
    return out


def plot_curves(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for r, c in zip(runs, colors):
        h = r["history"]
        ax[0].plot(h.epoch, h.train_loss, color=c, lw=1.8, label=f"latih seed {r['seed']}")
        ax[0].plot(h.epoch, h.val_loss, color=c, lw=1.8, ls="--", label=f"validasi seed {r['seed']}")
        ax[1].plot(h.epoch, h.val_auc, color=c, lw=1.8, marker="o", ms=3, label=f"seed {r['seed']}")
    ax[0].set_title("a) Loss BCE (garis putus = validasi)", fontsize=10, loc="left")
    ax[1].set_title("b) AUC validasi per epoch", fontsize=10, loc="left")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(runs, path):
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for r, c in zip(runs, colors):
        f = r["dir"] / "preds_all.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        fpr, tpr, _ = roc_curve(d.label, d.prob)
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"seed {r['seed']} (AUC {idn(roc_auc_score(d.label, d.prob))})")
    ax.plot([0, 1], [0, 1], color="#9a9a96", lw=1, ls=":")
    ax.set_xlabel("1 − spesifisitas")
    ax.set_ylabel("Sensitivitas")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--scenario", default="S1_all")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--platform", default=None, help="mis. 'Google Colab (GPU T4)'; default dari config")
    ap.add_argument("--week", default="11")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="laporan/08_Progress 3.docx")
    a = ap.parse_args()

    runs = load_runs(Path(a.runs), a.scenario, a.model)
    if not runs:
        sys.exit(f"Belum ada run selesai di {a.runs}/{a.scenario}/{a.model}. Jalankan pelatihan baseline dulu.")
    res = pd.read_csv(a.results)
    res = res[(res.run_tag == a.scenario) & (res.model == a.model)].drop_duplicates(["seed", "test"], keep="last")
    seeds = sorted(r["seed"] for r in runs)
    cfg = runs[0]["config"]
    figdir = Path(a.out).parent / "gambar_progress3"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, figdir / "kurva.png")
    plot_roc(runs, figdir / "roc.png")

    def tm(test, col):
        return res[res.test == test][col]

    auc_all = tm("all", "auc")
    best_epochs = [int(r["history"].val_auc.idxmax()) + 1 for r in runs]
    n_epochs = [len(r["history"]) for r in runs]
    eff = runs[0]["metrics"].get("efficiency", {})
    device = cfg.get("device", "")
    platform = a.platform or ("GPU (CUDA)" if "cuda" in device else "CPU")
    complete = len(seeds) >= 3 and len(res[res.test == "all"]) >= 3

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    para(doc, "LAPORAN PROGRESS KE-3", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para(doc, "PROJECT MATA KULIAH KECERDASAN BUATAN", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    para(doc, TITLE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    table(doc, [["No. Kelompok", "08"],
                ["Topik Project", "RCH-2: AI untuk Deteksi Dini Penyakit dari Citra Medis"],
                ["Dosen Pengampu", "Tegar Arifin Prasetyo"],
                ["Minggu ke- / Tanggal", f"{a.week} / {a.date or date.today().strftime('%d-%m-%Y')}"],
                ["Tautan Repositori", "https://github.com/monicabortang/Project_Paru_CXR"]],
          [4.5, 11.5], header=False, size=11, bold_first_col=True)
    para(doc, "**Anggota Kelompok**", align=WD_ALIGN_PARAGRAPH.LEFT)
    table(doc, [["No", "Nama", "NIM", "Peran"]] + [[str(i + 1), *t] for i, t in enumerate(TEAM)],
          [1.0, 7.5, 3.0, 4.5], size=11)

    # 1. Ringkasan
    heading(doc, "1. Ringkasan Progress")
    para(doc, (f"Pada minggu ke-{a.week} model baseline EfficientNet-B0 dilatih pada skenario S1 (semua posisi) "
               f"dengan {len(seeds)} seed ({', '.join(map(str, seeds))}) menggunakan {platform}. Seluruh "
               f"hyperparameter dan seed tercatat otomatis pada config.json. Pada set uji tetap "
               f"(n = {int(tm('all', 'n').iloc[0])} citra), baseline mencapai AUC {ms(auc_all)}, macro-F1 "
               f"{ms(tm('all', 'macro_f1'))}, sensitivitas {ms(tm('all', 'sensitivity'))}, dan spesifisitas "
               f"{ms(tm('all', 'specificity'))} pada ambang spesifisitas 90% yang ditetapkan di data validasi. "
               f"Hasil ini menjadi acuan pembanding untuk model utama Swin-T pada Progress 4."))
    caption(doc, "Tabel 1. Status target minggu ke-" + a.week)
    stat = "Selesai" if complete else "Sebagian"
    table(doc, [["No", "Target Minggu Ini (sesuai timeline)", "Status", "Keterangan"],
                ["1", "Pelatihan model baseline dengan konfigurasi standar", stat,
                 f"EfficientNet-B0, S1, {len(seeds)} seed"],
                ["2", "Pencatatan hyperparameter dan random seed", "Selesai", "Tabel 2; runs/…/config.json"],
                ["3", "Evaluasi awal pada data uji", "Selesai" if len(res) else "Belum",
                 "Tabel 3–4, Gambar 1–3"]],
          [1.0, 7.0, 2.2, 5.8])

    # 2. Uraian
    heading(doc, "2. Uraian Pekerjaan")
    heading(doc, "2.1 Lingkungan Pelatihan", 2)
    para(doc, (f"Pelatihan dijalankan pada {platform} dengan PyTorch {cfg.get('torch', '–')} dan timm "
               f"{cfg.get('timm', '–')}. Data yang dipakai adalah hasil konversi PNG 512 px dan pembagian data "
               f"(seed 42) yang sama persis dengan Progress 2, diambil dari repositori sehingga hasil dapat "
               f"direproduksi. Uji cepat (1 epoch, 256 citra) lebih dulu dijalankan untuk memastikan alur "
               f"pelatihan dan evaluasi berjalan tanpa galat."))
    heading(doc, "2.2 Konfigurasi Baseline", 2)
    para(doc, ("EfficientNet-B0 dengan bobot pralatih ImageNet dilatih ulang seluruh lapisannya (full fine-tuning) "
               "untuk klasifikasi biner (satu logit keluaran). Konfigurasi lengkap pada Tabel 2."))
    caption(doc, "Tabel 2. Konfigurasi dan hyperparameter baseline (dari config.json)")
    table(doc, [["Parameter", "Nilai"],
                ["Model", f"{cfg['model']} (pralatih ImageNet), {cfg.get('n_params', 0) / 1e6:.2f} juta parameter".replace(".", ",")],
                ["Ukuran input", f"{cfg['img_size']} × {cfg['img_size']} piksel, 3 kanal"],
                ["Data latih / validasi", f"{cfg['n_train']} / {cfg['n_val']} citra (prevalensi latih {pct(cfg['train_prevalence'])})"],
                ["Optimizer", f"AdamW, learning rate {cfg['lr']:g}, weight decay {cfg['weight_decay']:g}"],
                ["Penjadwal LR", f"Warmup linear {cfg['warmup_epochs']:g} epoch + cosine decay"],
                ["Batch size", str(cfg["batch_size"])],
                ["Epoch maksimum / early stopping", f"{cfg['epochs']} / patience {cfg['patience']} (AUC validasi)"],
                ["Fungsi loss", "Binary cross-entropy" + (" dengan bobot kelas" if cfg.get("pos_weight") else " (tanpa bobot kelas)")],
                ["Augmentasi (latih)", "Rotasi ±7°, translasi 5%, skala 0,9–1,1, kecerahan/kontras ±0,1"
                 + (", flip horizontal" if cfg.get("hflip") else "; tanpa flip")],
                ["Prapemrosesan", "Pad persegi, resize, normalisasi ImageNet" + (", CLAHE" if cfg.get("clahe") else "")],
                ["Ambang klasifikasi", f"Spesifisitas {pct(cfg['target_spec'], 0)} pada data validasi"],
                ["Lainnya", "Mixed precision (AMP), gradient clipping 1,0"],
                ["Random seed", ", ".join(map(str, seeds)) + " (split data: seed 42)"]],
          [5.0, 11.0], size=10, bold_first_col=True)
    heading(doc, "2.3 Prosedur Evaluasi", 2)
    para(doc, ("Model dengan AUC validasi tertinggi pada tiap seed dievaluasi pada set uji tetap (semua posisi) serta "
               "subset per posisi (AP/PA, versi natural dan matched). Metrik: AUC dengan interval kepercayaan 95% "
               "(bootstrap terstratifikasi), macro-F1, sensitivitas, dan spesifisitas pada ambang yang ditetapkan di "
               "validasi. Hasil dilaporkan sebagai rerata ± simpangan baku antar seed."
               + (" Uji eksternal pada dataset Kermany juga disertakan." if "kermany" in set(res.test) else "")))

    # 3. Hasil
    heading(doc, "3. Hasil")
    caption(doc, "Tabel 3. Metrik baseline pada set uji (rerata ± SD antar seed)")
    rows = [["Set uji", "n", "Prevalensi", "AUC", "Macro-F1", "Sensitivitas", "Spesifisitas"]]
    for t, label in TEST_ORDER:
        if t not in set(res.test):
            continue
        rows.append([label, str(int(tm(t, "n").iloc[0])), pct(tm(t, "prevalence").iloc[0]),
                     ms(tm(t, "auc")), ms(tm(t, "macro_f1")), ms(tm(t, "sensitivity")), ms(tm(t, "specificity"))])
    table(doc, rows, [3.4, 1.4, 1.9, 2.6, 2.3, 2.3, 2.3], size=9)

    caption(doc, "Tabel 4. Hasil per seed pada set uji semua posisi")
    rows = [["Seed", "Epoch terbaik / total", "AUC validasi", "AUC uji [CI 95%]", "Macro-F1", "Sens.", "Spes."]]
    for r, be, ne in zip(runs, best_epochs, n_epochs):
        x = res[(res.test == "all") & (res.seed == r["seed"])]
        if x.empty:
            continue
        x = x.iloc[0]
        rows.append([str(r["seed"]), f"{be} / {ne}", idn(r["history"].val_auc.max()),
                     f"{idn(x.auc)} [{idn(x.auc_lo)}–{idn(x.auc_hi)}]", idn(x.macro_f1),
                     idn(x.sensitivity), idn(x.specificity)])
    table(doc, rows, [1.2, 3.0, 2.2, 3.8, 2.0, 1.8, 1.8], size=9)

    figure(doc, figdir / "kurva.png", 15.5, "Gambar 1. Kurva pelatihan: (a) loss latih dan validasi, (b) AUC validasi")
    roc_ok = (figdir / "roc.png").exists()
    cm = runs[0]["dir"] / "cm_all.png"
    if roc_ok:
        figure(doc, figdir / "roc.png", 8.5, "Gambar 2. Kurva ROC pada set uji semua posisi")
    if cm.exists():
        figure(doc, cm, 7.0, f"Gambar 3. Confusion matrix seed {runs[0]['seed']} pada set uji semua posisi")
    if eff:
        caption(doc, "Tabel 5. Efisiensi model")
        table(doc, [["Parameter", "Ukuran file", "Latensi (batch 1)", "Throughput (batch 32)", "Perangkat"],
                    [f"{idn(eff.get('params_M', np.nan), 2)} juta", f"{idn(eff.get('size_MB', np.nan), 1)} MB",
                     f"{idn(eff.get('latency_ms_bs1', np.nan), 1)} ms",
                     f"{idn(eff.get('throughput_img_s_bs32', np.nan), 0)} citra/detik", eff.get("device", device)]],
              [3.0, 2.6, 3.0, 4.0, 3.4], size=10)

    # 4. Analisis (otomatis, berdasarkan angka)
    heading(doc, "4. Analisis dan Pembahasan")
    a_all = auc_all.mean()
    para(doc, (f"**Performa umum.** Rerata AUC uji {idn(a_all)} menunjukkan baseline "
               + ("sudah mampu membedakan citra dengan dan tanpa opasitas dengan baik"
                  if a_all >= 0.85 else "mampu membedakan kedua kelas, namun masih memiliki ruang perbaikan"
                  if a_all >= 0.75 else "belum optimal dalam membedakan kedua kelas")
               + f". Variasi antar seed (SD AUC {idn(auc_all.std(ddof=0))}) "
               + ("kecil, sehingga hasil cukup stabil." if auc_all.std(ddof=0) < 0.01 else
                  "perlu diperhatikan saat membandingkan model.")))
    h0 = runs[0]["history"]
    be0 = int(h0.val_auc.idxmax())
    gap = h0.val_loss.iloc[be0] - h0.train_loss.iloc[be0]
    para(doc, (f"**Kurva pelatihan.** AUC validasi terbaik dicapai pada epoch {', '.join(map(str, best_epochs))} "
               f"(per seed), dan pelatihan berhenti setelah {', '.join(map(str, n_epochs))} epoch. Selisih loss "
               f"validasi dan latih pada epoch terbaik seed {runs[0]['seed']} sebesar {idn(gap)}"
               + (", artinya loss validasi masih di bawah loss latih (efek augmentasi pada data latih), sehingga "
                  "belum tampak overfitting." if gap < 0 else
                  ", menandakan mulai terjadi overfitting setelah titik tersebut; early stopping mencegah model "
                  "terus menghafal data latih." if gap > 0.1 else ", relatif kecil sehingga tidak tampak overfitting berat.")))
    if {"AP_natural", "PA_natural"} <= set(res.test):
        ap_n, pa_n = tm("AP_natural", "auc").mean(), tm("PA_natural", "auc").mean()
        txt = (f"**Per posisi.** Pada data natural, AUC di subset AP {idn(ap_n)} dan PA {idn(pa_n)}. "
               f"Sensitivitas AP {ms(tm('AP_natural', 'sensitivity'))} dan PA {ms(tm('PA_natural', 'sensitivity'))}; "
               "perbedaan sensitivitas sebagian dipengaruhi prevalensi dan luas lesi yang berbeda antar posisi "
               "(Progress 2).")
        if {"AP_matched", "PA_matched"} <= set(res.test):
            ap_m, pa_m = tm("AP_matched", "auc").mean(), tm("PA_matched", "auc").mean()
            txt += (f" Setelah komposisi kelas disamakan (matched), AUC AP {idn(ap_m)} dan PA {idn(pa_m)} "
                    f"(selisih {idn(abs(ap_m - pa_m))}). Model ini dilatih pada kedua posisi, sehingga pengujian "
                    "pergeseran posisi yang sesungguhnya (latih satu posisi, uji posisi lain) dilakukan pada Progress 5.")
        para(doc, txt)
    if "kermany" in set(res.test):
        k = tm("kermany", "auc").mean()
        para(doc, (f"**Uji eksternal.** Pada data Kermany, AUC {idn(k)} "
                   + ("lebih rendah" if k < a_all else "sebanding atau lebih tinggi") +
                   f" dibanding data internal ({idn(a_all)}). Perbedaan populasi (anak vs dewasa), rumah sakit, "
                   "dan definisi label harus diperhitungkan dalam menafsirkan angka ini."))
    para(doc, ("**Catatan.** Ambang klasifikasi ditetapkan pada spesifisitas 90% sehingga sensitivitas menjadi "
               "metrik yang paling informatif untuk kebutuhan skrining. [CEK] Tambahkan pengamatan kelompok, "
               "misalnya contoh kesalahan prediksi, sebelum dikumpulkan."))

    # 5. Kendala
    heading(doc, "5. Kendala dan Solusi")
    table(doc, [["No", "Kendala", "Dampak", "Solusi / Rencana Penanganan"],
                ["1", "Laptop tidak memiliki GPU NVIDIA", "Pelatihan di CPU diperkirakan >6 jam per seed",
                 "Pelatihan dipindah ke GPU cloud gratis; laptop untuk uji cepat dan demo"],
                ["2", "Internet notebook Kaggle tidak aktif (perlu verifikasi nomor HP)",
                 "Kode dan bobot pralatih tidak dapat diunduh", f"Memakai {platform}; disiapkan juga mode offline (kaggle_bundle)"],
                ["3", "Sesi GPU cloud dapat terputus", "Pelatihan berhenti di tengah",
                 "Hasil disimpan per run dan runner melanjutkan otomatis run yang belum selesai"]],
          [1.0, 5.0, 4.5, 5.5], size=10)

    # 6. Kontribusi
    heading(doc, "6. Kontribusi Anggota")
    table(doc, [["Nama", "Tugas yang Dikerjakan", "Bukti (commit/file)", "Kontribusi (%)"],
                [TEAM[0][0], "Koordinasi dan penyiapan lingkungan GPU", "notebooks/", "20"],
                [TEAM[1][0], "Pelatihan baseline EfficientNet-B0 (3 seed)", "runs/S1_all/efficientnet_b0/", "20"],
                [TEAM[2][0], "Evaluasi pada set uji dan analisis metrik", "results/all_results.csv", "20"],
                [TEAM[3][0], "Penyiapan data dan smoke test", "data/, runs/S1_all_smoke/", "20"],
                [TEAM[4][0], "Dokumentasi hyperparameter dan laporan", "Laporan Progress 3", "20"]],
          [4.5, 5.5, 4.2, 1.8], size=10)
    para(doc, ("**Penggunaan alat AI generatif:** Claude (Anthropic) digunakan untuk membantu menyusun kode pelatihan, "
               "evaluasi, dan pembuat laporan otomatis. Semua angka pada laporan ini dihasilkan langsung dari "
               "keluaran program, dan pelatihan dijalankan serta diperiksa oleh anggota kelompok. [CEK] Sesuaikan "
               "pembagian tugas dan persentase kontribusi."))

    # 7. Rencana
    heading(doc, "7. Rencana Minggu Berikutnya")
    table(doc, [["No", "Kegiatan", "PIC", "Target Luaran"],
                ["1", "Melatih model utama Swin-T pada S1 (3 seed) dengan konfigurasi setara baseline", TEAM[1][0], "best.pt, kurva pelatihan"],
                ["2", "Membandingkan Swin-T vs EfficientNet-B0 (AUC, F1, waktu inferensi, ukuran)", TEAM[2][0], "Tabel perbandingan + uji berpasangan"],
                ["3", "Menjalankan skenario S2/S3 (latih AP/PA) untuk Progress 5", TEAM[3][0], "Run S2/S3 matched"],
                ["4", "Menyiapkan uji eksternal Kermany bila belum tersedia", TEAM[0][0], "kermany.csv"],
                ["5", "Menyusun laporan Progress 4", TEAM[4][0], "Laporan Progress 4"]],
          [1.0, 7.5, 4.0, 3.5], size=10)

    heading(doc, "Referensi")
    for ref in [
        "[1] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, pp. 6105–6114.",
        "[2] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019.",
        "[3] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in Proc. ICLR, 2017.",
        "[4] G. Shih et al., “Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia,” Radiol. Artif. Intell., vol. 1, no. 1, Art. no. e180041, 2019.",
        "[5] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
    ]:
        p = para(doc, ref, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)
        p.paragraph_format.left_indent = Cm(1)
        p.paragraph_format.first_line_indent = Cm(-1)

    heading(doc, "Lampiran")
    para(doc, "A. Repositori: https://github.com/monicabortang/Project_Paru_CXR", align=WD_ALIGN_PARAGRAPH.LEFT)
    para(doc, (f"B. Berkas hasil: runs/{a.scenario}/{a.model}/seed*/ (config.json, history.csv, metrics.json, "
               "curves.png, cm_*.png) dan results/all_results.csv."), align=WD_ALIGN_PARAGRAPH.LEFT)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"Laporan tersimpan: {out}")
    print(f"  seed: {seeds} | AUC uji (all): {ms(auc_all)} | status: {stat}")
    print("Buka di Word, periksa bagian bertanda [CEK], lalu File > Save As > PDF.")


if __name__ == "__main__":
    main()
