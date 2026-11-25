"""Membuat Laporan Progress 4 (.docx) OTOMATIS: model utama Swin-T vs baseline EfficientNet-B0.

Semua angka diambil dari hasil run asli (runs/, results/all_results.csv,
results/tables/tabel_uji_berpasangan.csv). Jalankan setelah grup 'baseline' dan 'main'
selesai dan setelah `python -m src.summarize`.

  python scripts/make_progress4_report.py --platform "Google Colab (GPU NVIDIA T4)"

Perlu: pip install python-docx matplotlib pandas scikit-learn
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_progress3_report import (TEAM, TEST_ORDER, TITLE, Cm, Document, Pt,  # noqa: E402
                                   WD_ALIGN_PARAGRAPH, caption, figure, heading, idn,
                                   load_runs, ms, para, pct, qn, table)

BASE, MAIN = "efficientnet_b0", "swin_tiny_patch4_window7_224"
NAME = {BASE: "EfficientNet-B0", MAIN: "Swin-T"}
COL = {BASE: "#2a78d6", MAIN: "#eb6834"}


def plot_curves(runs, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    colors = ["#eb6834", "#2a78d6", "#1baf7a"]
    for r, c in zip(runs, colors):
        h = r["history"]
        ax[0].plot(h.epoch, h.train_loss, color=c, lw=1.8, label=f"latih seed {r['seed']}")
        ax[0].plot(h.epoch, h.val_loss, color=c, lw=1.8, ls="--", label=f"validasi seed {r['seed']}")
        ax[1].plot(h.epoch, h.val_auc, color=c, lw=1.8, marker="o", ms=3, label=f"seed {r['seed']}")
    ax[0].set_title(f"a) Loss BCE {title} (putus = validasi)", fontsize=10, loc="left")
    ax[1].set_title(f"b) AUC validasi {title}", fontsize=10, loc="left")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ensemble(runs_dir: Path, model: str, test: str):
    files = sorted((runs_dir / "S1_all" / model).glob(f"seed*/preds_{test}.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f)[["patientId", "label", "prob"]] for f in files]
    out = dfs[0].copy()
    out["prob"] = np.mean([d.set_index("patientId").loc[out.patientId, "prob"].values for d in dfs], axis=0)
    return out


def plot_roc(runs_dir: Path, path):
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    for m in (BASE, MAIN):
        d = ensemble(runs_dir, m, "all")
        if d is None:
            continue
        fpr, tpr, _ = roc_curve(d.label, d.prob)
        ax.plot(fpr, tpr, color=COL[m], lw=1.9,
                label=f"{NAME[m]} (AUC {idn(roc_auc_score(d.label, d.prob))})")
    ax.plot([0, 1], [0, 1], color="#9a9a96", lw=1, ls=":")
    ax.set_xlabel("1 − spesifisitas")
    ax.set_ylabel("Sensitivitas")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bars(res, tests, path):
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    x = np.arange(len(tests))
    w = 0.36
    for i, m in enumerate((BASE, MAIN)):
        g = res[res.model == m].groupby("test").auc
        mean = [g.mean().get(t, np.nan) for t, _ in tests]
        sd = [g.std(ddof=0).get(t, 0) for t, _ in tests]
        b = ax.bar(x + (i - .5) * (w + .02), mean, w, yerr=sd, color=COL[m], label=NAME[m],
                   capsize=3, error_kw=dict(lw=1, ecolor="#5c5c58"), zorder=3)
        for r, v in zip(b, mean):
            if not np.isnan(v):
                ax.text(r.get_x() + r.get_width() / 2, v + .012, idn(v), ha="center", fontsize=7.5)
    ax.set_xticks(x, [lab for _, lab in tests], fontsize=8.5)
    lo = np.nanmin(res.auc) - .08
    ax.set_ylim(max(0.5, lo), 1.0)
    ax.set_ylabel("AUC (rerata ± SD)")
    ax.yaxis.grid(True, alpha=.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--paired", default="results/tables/tabel_uji_berpasangan.csv")
    ap.add_argument("--platform", default="GPU (CUDA)")
    ap.add_argument("--week", default="12")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="laporan/08_Progress 4.docx")
    a = ap.parse_args()

    runs_dir = Path(a.runs)
    rb, rm = load_runs(runs_dir, "S1_all", BASE), load_runs(runs_dir, "S1_all", MAIN)
    if not rm:
        sys.exit("Belum ada run Swin-T selesai (runs/S1_all/swin_tiny_patch4_window7_224). Jalankan grup 'main'.")
    if not rb:
        sys.exit("Run baseline EfficientNet-B0 tidak ditemukan. Jalankan grup 'baseline' dulu.")
    res = pd.read_csv(a.results)
    res = res[(res.run_tag == "S1_all") & res.model.isin([BASE, MAIN])].drop_duplicates(
        ["model", "seed", "test"], keep="last")
    tests = [(t, lab) for t, lab in TEST_ORDER if t in set(res.test)]
    cfg_b, cfg_m = rb[0]["config"], rm[0]["config"]
    seeds = sorted(r["seed"] for r in rm)

    figdir = Path(a.out).parent / "gambar_progress4"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_curves(rm, figdir / "kurva_swin.png", "Swin-T")
    plot_roc(runs_dir, figdir / "roc_banding.png")
    plot_bars(res, tests, figdir / "auc_banding.png")

    def col(m, t, c):
        return res[(res.model == m) & (res.test == t)][c]

    paired = pd.DataFrame()
    if Path(a.paired).exists():
        p = pd.read_csv(a.paired)
        paired = p[(p.skenario == "S1_all") & p.perbandingan.str.startswith(BASE)]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    para(doc, "LAPORAN PROGRESS KE-4", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
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

    ab, am = col(BASE, "all", "auc"), col(MAIN, "all", "auc")
    d_all = am.mean() - ab.mean()
    pr_all = paired[paired.uji == "all"]
    sig_txt = ""
    if len(pr_all):
        x = pr_all.iloc[0]
        lo, hi = -x.ci_hi, -x.ci_lo
        sig = "bermakna secara statistik" if (lo > 0 or hi < 0) else "belum bermakna secara statistik"
        sig_txt = f" (uji bootstrap berpasangan: ΔAUC {idn(-x.dAUC)}, CI 95% [{idn(lo)}; {idn(hi)}], {sig})"

    heading(doc, "1. Ringkasan Progress")
    para(doc, (f"Pada minggu ke-{a.week} model utama Swin-T diimplementasikan dan dilatih pada skenario S1 dengan "
               f"pembagian data, prapemrosesan, augmentasi, dan seed ({', '.join(map(str, seeds))}) yang sama dengan "
               f"baseline, menggunakan {a.platform}. Pada set uji semua posisi, Swin-T mencapai AUC {ms(am)} "
               f"dibanding EfficientNet-B0 {ms(ab)}; selisih {idn(d_all)}{sig_txt}. Swin-T memiliki "
               f"{idn(cfg_m['n_params'] / 1e6, 1)} juta parameter, sekitar {cfg_m['n_params'] / cfg_b['n_params']:.0f} "
               f"kali baseline. Hasil ini menjadi dasar skenario pergeseran posisi AP/PA pada Progress 5."))
    caption(doc, f"Tabel 1. Status target minggu ke-{a.week}")
    done = len(seeds) >= 3
    table(doc, [["No", "Target Minggu Ini (sesuai timeline)", "Status", "Keterangan"],
                ["1", "Implementasi model utama Swin-T", "Selesai", "src/models.py (timm)"],
                ["2", "Pelatihan dengan pembagian data yang sama dengan baseline", "Selesai" if done else "Sebagian",
                 f"S1, {len(seeds)} seed"],
                ["3", "Tabel perbandingan awal baseline vs model utama", "Selesai", "Tabel 3–5, Gambar 2–3"]],
          [1.0, 7.0, 2.2, 5.8])

    heading(doc, "2. Uraian Pekerjaan")
    heading(doc, "2.1 Arsitektur Model Utama", 2)
    para(doc, ("Swin Transformer [1] memecah citra menjadi patch 4×4 piksel lalu menghitung self-attention di dalam "
               "jendela lokal berukuran 7×7 patch. Jendela digeser (shifted window) pada blok berikutnya sehingga "
               "informasi dapat mengalir antarjendela. Arsitektur bersifat hierarkis dengan 4 tahap (jumlah blok "
               "2-2-6-2 dan dimensi awal 96 untuk varian Tiny); resolusi fitur mengecil dan jumlah kanal bertambah "
               "di setiap tahap, mirip CNN. Kepala klasifikasi asli diganti dengan satu keluaran (logit) untuk "
               "klasifikasi biner. Berbeda dengan EfficientNet-B0 [2] yang mengandalkan konvolusi lokal, Swin-T dapat "
               "memodelkan hubungan antarbagian citra yang lebih jauh, yang diharapkan membantu mengenali pola "
               "opasitas yang tersebar."))
    heading(doc, "2.2 Konfigurasi Pelatihan", 2)
    caption(doc, "Tabel 2. Konfigurasi baseline dan model utama (dari config.json)")
    table(doc, [["Parameter", "EfficientNet-B0 (baseline)", "Swin-T (model utama)"],
                ["Jumlah parameter", f"{idn(cfg_b['n_params'] / 1e6, 2)} juta", f"{idn(cfg_m['n_params'] / 1e6, 2)} juta"],
                ["Bobot awal", "ImageNet", "ImageNet"],
                ["Ukuran input", f"{cfg_b['img_size']}×{cfg_b['img_size']}", f"{cfg_m['img_size']}×{cfg_m['img_size']}"],
                ["Learning rate (AdamW)", f"{cfg_b['lr']:g}", f"{cfg_m['lr']:g}"],
                ["Batch size", str(cfg_b["batch_size"]), str(cfg_m["batch_size"])],
                ["Weight decay", f"{cfg_b['weight_decay']:g}", f"{cfg_m['weight_decay']:g}"],
                ["Epoch maks. / patience", f"{cfg_b['epochs']} / {cfg_b['patience']}", f"{cfg_m['epochs']} / {cfg_m['patience']}"],
                ["Data latih / validasi", f"{cfg_b['n_train']} / {cfg_b['n_val']}", f"{cfg_m['n_train']} / {cfg_m['n_val']}"],
                ["Penjadwal, augmentasi, ambang", "Sama untuk kedua model", "Sama untuk kedua model"],
                ["Seed", ", ".join(str(r["seed"]) for r in rb), ", ".join(map(str, seeds))]],
          [5.0, 5.5, 5.5], size=10, bold_first_col=True)
    para(doc, ("Learning rate Swin-T dibuat lebih kecil (praktik umum fine-tuning transformer) dan batch size "
               "disesuaikan dengan memori GPU; selain itu seluruh pengaturan identik agar perbandingan adil."))
    heading(doc, "2.3 Prosedur Perbandingan", 2)
    para(doc, ("Kedua model dievaluasi pada set uji yang sama. Metrik dilaporkan sebagai rerata ± SD dari 3 seed. "
               "Perbedaan AUC diuji dengan bootstrap berpasangan [4] pada prediksi rata-rata antar seed (citra uji "
               "identik), sehingga dapat diketahui apakah selisih performa bermakna atau hanya variasi acak."))

    heading(doc, "3. Hasil")
    figure(doc, figdir / "kurva_swin.png", 15.5, "Gambar 1. Kurva pelatihan Swin-T: (a) loss, (b) AUC validasi")
    caption(doc, "Tabel 3. Perbandingan AUC dan macro-F1 per set uji (rerata ± SD)")
    rows = [["Set uji", "AUC EffNet-B0", "AUC Swin-T", "ΔAUC", "F1 EffNet-B0", "F1 Swin-T"]]
    for t, lab in tests:
        rows.append([lab, ms(col(BASE, t, "auc")), ms(col(MAIN, t, "auc")),
                     idn(col(MAIN, t, "auc").mean() - col(BASE, t, "auc").mean()),
                     ms(col(BASE, t, "macro_f1")), ms(col(MAIN, t, "macro_f1"))])
    table(doc, rows, [3.2, 2.8, 2.8, 1.6, 2.8, 2.8], size=9)
    caption(doc, "Tabel 4. Sensitivitas dan spesifisitas pada ambang spesifisitas 90% (validasi)")
    rows = [["Set uji", "Sens. EffNet-B0", "Sens. Swin-T", "Spes. EffNet-B0", "Spes. Swin-T"]]
    for t, lab in tests:
        rows.append([lab, ms(col(BASE, t, "sensitivity")), ms(col(MAIN, t, "sensitivity")),
                     ms(col(BASE, t, "specificity")), ms(col(MAIN, t, "specificity"))])
    table(doc, rows, [3.4, 3.1, 3.1, 3.1, 3.1], size=9)
    figure(doc, figdir / "auc_banding.png", 15.5, "Gambar 2. AUC per set uji: EfficientNet-B0 vs Swin-T")
    if len(paired):
        caption(doc, "Tabel 5. Uji bootstrap berpasangan ΔAUC (Swin-T − EfficientNet-B0)")
        rows = [["Set uji", "ΔAUC", "CI 95%", "p-value", "Kesimpulan"]]
        lab_of = dict(tests)
        for t, lab in tests:
            x = paired[paired.uji == t]
            if x.empty:
                continue
            x = x.iloc[0]
            lo, hi = -x.ci_hi, -x.ci_lo
            concl = ("Swin-T lebih baik" if lo > 0 else "EfficientNet-B0 lebih baik" if hi < 0
                     else "Tidak berbeda bermakna")
            rows.append([lab_of[t], idn(-x.dAUC), f"[{idn(lo)}; {idn(hi)}]", idn(x.p), concl])
        table(doc, rows, [3.6, 2.0, 3.6, 2.0, 4.8], size=9)
    figure(doc, figdir / "roc_banding.png", 8.5, "Gambar 3. Kurva ROC (prediksi rata-rata 3 seed) pada set uji semua posisi")
    eb = rb[0]["metrics"].get("efficiency", {})
    em = rm[0]["metrics"].get("efficiency", {})
    if eb and em:
        caption(doc, "Tabel 6. Efisiensi model")
        table(doc, [["Model", "Parameter", "Ukuran file", "Latensi (batch 1)", "Throughput (batch 32)"],
                    *[[NAME[m], f"{idn(e['params_M'], 2)} juta", f"{idn(e['size_MB'], 1)} MB",
                       f"{idn(e['latency_ms_bs1'], 1)} ms", f"{idn(e['throughput_img_s_bs32'], 0)} citra/detik"]
                      for m, e in ((BASE, eb), (MAIN, em))]],
              [3.4, 2.6, 2.6, 3.4, 4.0], size=10)

    heading(doc, "4. Analisis dan Pembahasan")
    para(doc, (f"**Performa keseluruhan.** Pada set uji semua posisi, Swin-T "
               + ("lebih unggul" if d_all > 0.005 else "lebih rendah" if d_all < -0.005 else "setara")
               + f" dibanding baseline (ΔAUC {idn(d_all)}){sig_txt}. "
               + ("Keunggulan ini konsisten dengan kemampuan attention menangkap pola opasitas yang tersebar."
                  if d_all > 0.005 else
                  "Dengan data latih ±19 ribu citra, kelebihan transformer belum terlihat; CNN ringan sudah kompetitif."
                  if d_all <= 0.005 else "")))
    if {"AP_natural", "PA_natural"} <= set(res.test):
        txt = "**Per posisi.** "
        for m in (BASE, MAIN):
            txt += (f"{NAME[m]}: AUC AP {idn(col(m, 'AP_natural', 'auc').mean())}, PA "
                    f"{idn(col(m, 'PA_natural', 'auc').mean())}, sensitivitas PA "
                    f"{idn(col(m, 'PA_natural', 'sensitivity').mean())}. ")
        txt += ("Pada kedua model, sensitivitas di citra PA jauh lebih rendah meskipun AUC-nya tinggi. Ini menunjukkan "
                "ambang keputusan yang ditetapkan dari data campuran tidak cocok untuk citra PA yang lesinya lebih kecil "
                "dan prevalensinya rendah; kalibrasi ambang per domain akan diuji pada Progress 5.")
        para(doc, txt)
    if "kermany" in set(res.test):
        para(doc, (f"**Uji eksternal.** Pada Kermany, AUC EfficientNet-B0 {ms(col(BASE, 'kermany', 'auc'))} dan Swin-T "
                   f"{ms(col(MAIN, 'kermany', 'auc'))}, dengan sensitivitas {idn(col(BASE, 'kermany', 'sensitivity').mean())} "
                   f"vs {idn(col(MAIN, 'kermany', 'sensitivity').mean())}. Pola yang sama (AUC tinggi, sensitivitas rendah) "
                   "menunjukkan pergeseran kalibrasi probabilitas saat model dipakai di rumah sakit lain."))
    if eb and em:
        para(doc, (f"**Efisiensi.** Swin-T {em['params_M'] / eb['params_M']:.0f}× lebih besar dan latensinya "
                   f"{idn(em['latency_ms_bs1'], 1)} ms vs {idn(eb['latency_ms_bs1'], 1)} ms per citra. Untuk aplikasi "
                   "mobile/skrining, selisih performa perlu dibandingkan dengan biaya komputasi ini. "
                   "[CEK] Tambahkan pengamatan kelompok sebelum dikumpulkan."))

    heading(doc, "5. Kendala dan Solusi")
    table(doc, [["No", "Kendala", "Dampak", "Solusi / Rencana Penanganan"],
                ["1", "Swin-T lebih berat daripada baseline", "Waktu latih per seed lebih lama",
                 "Mixed precision (AMP) dan batch size disesuaikan memori GPU T4"],
                ["2", "Sesi Colab gratis dapat terputus", "Pelatihan berhenti di tengah",
                 "Hasil disimpan di Google Drive; run yang selesai dilewati, cache data PNG di Drive"],
                ["3", "Sensitivitas rendah pada PA dan data eksternal", "Banyak kasus positif terlewat pada ambang tetap",
                 "Diuji kalibrasi ambang per domain dan skenario S2/S3 pada Progress 5"]],
          [1.0, 5.0, 4.5, 5.5], size=10)

    heading(doc, "6. Kontribusi Anggota")
    table(doc, [["Nama", "Tugas yang Dikerjakan", "Bukti (commit/file)", "Kontribusi (%)"],
                [TEAM[0][0], "Koordinasi dan menjalankan pelatihan di Colab", "notebooks/colab_progress4.ipynb", "20"],
                [TEAM[1][0], "Implementasi dan konfigurasi Swin-T", "src/models.py; runs/S1_all/swin…", "20"],
                [TEAM[2][0], "Perbandingan metrik dan uji berpasangan", "results/tables/", "20"],
                [TEAM[3][0], "Pemantauan data dan efisiensi model", "metrics.json", "20"],
                [TEAM[4][0], "Penyusunan laporan Progress 4", "Laporan Progress 4", "20"]],
          [4.5, 5.5, 4.2, 1.8], size=10)
    para(doc, ("**Penggunaan alat AI generatif:** Claude (Anthropic) digunakan untuk membantu menyusun kode pelatihan "
               "dan pembuat laporan otomatis. Semua angka berasal langsung dari keluaran program. [CEK] Sesuaikan "
               "pembagian tugas dan persentase kontribusi."))

    heading(doc, "7. Rencana Minggu Berikutnya")
    table(doc, [["No", "Kegiatan", "PIC", "Target Luaran"],
                ["1", "Skenario S2/S3 matched: latih AP uji PA dan sebaliknya (kedua model)", TEAM[1][0], "Tabel drop & gap AUC"],
                ["2", "Versi natural S2/S3 dan S4 (efisiensi data target)", TEAM[3][0], "Tabel hasil skenario"],
                ["3", "Ablasi CLAHE dan bobot kelas; kalibrasi ambang per domain", TEAM[2][0], "Tabel ablasi"],
                ["4", "Uji eksternal Kermany untuk semua skenario", TEAM[0][0], "Tabel S6"],
                ["5", "Menyusun laporan Progress 5", TEAM[4][0], "Laporan Progress 5"]],
          [1.0, 7.5, 4.0, 3.5], size=10)

    heading(doc, "Referensi")
    for ref in [
        "[1] Z. Liu et al., “Swin Transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF ICCV, 2021, pp. 10012–10022.",
        "[2] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, pp. 6105–6114.",
        "[3] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. ICLR, 2021.",
        "[4] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
    ]:
        p = para(doc, ref, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)
        p.paragraph_format.left_indent = Cm(1)
        p.paragraph_format.first_line_indent = Cm(-1)
    heading(doc, "Lampiran")
    para(doc, "A. Repositori: https://github.com/monicabortang/Project_Paru_CXR", align=WD_ALIGN_PARAGRAPH.LEFT)
    para(doc, ("B. Berkas hasil: runs/S1_all/swin_tiny_patch4_window7_224/seed*/, runs/S1_all/efficientnet_b0/seed*/, "
               "results/all_results.csv, results/tables/."), align=WD_ALIGN_PARAGRAPH.LEFT)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"Laporan tersimpan: {out}")
    print(f"  AUC uji (all): EffNet-B0 {ms(ab)} | Swin-T {ms(am)} | Δ {idn(d_all)}")


if __name__ == "__main__":
    main()
