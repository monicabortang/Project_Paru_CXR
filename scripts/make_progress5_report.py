"""Membuat Laporan Progress 5 (.docx) OTOMATIS: eksperimen inti pergeseran posisi AP<->PA,
versi natural, ablasi, tuning learning rate, kalibrasi ambang per domain, dan uji eksternal.

Semua angka diambil dari hasil run asli. Jalankan setelah grup baseline, main, p5a, p5b
dan setelah `python -m src.summarize`.

  python scripts/make_progress5_report.py --platform "Google Colab (GPU NVIDIA T4)"
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from make_progress3_report import (TEAM, TITLE, Cm, Document, Pt, WD_ALIGN_PARAGRAPH,  # noqa: E402
                                   caption, figure, heading, idn, ms, para, qn, table)
from src.common import binary_metrics, threshold_at_specificity  # noqa: E402

BASE, MAIN = "efficientnet_b0", "swin_tiny_patch4_window7_224"
MODELS = [BASE, MAIN]
NAME = {BASE: "EfficientNet-B0", MAIN: "Swin-T"}
SCEN = [("S1_all", "S1 (semua posisi)"), ("S2_AP_matched", "S2 latih AP (matched)"),
        ("S3_PA_matched", "S3 latih PA (matched)"), ("S2_AP_natural", "S2 latih AP (natural)"),
        ("S3_PA_natural", "S3 latih PA (natural)")]
TESTS = [("AP_matched", "AP matched"), ("PA_matched", "PA matched"), ("AP_natural", "AP natural"),
         ("PA_natural", "PA natural"), ("kermany", "Kermany")]


def heatmaps(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, m in zip(axes, MODELS):
        mat = np.full((2, 2), np.nan)
        for i, s in enumerate(["S2_AP_matched", "S3_PA_matched"]):
            for j, t in enumerate(["AP_matched", "PA_matched"]):
                v = res[(res.model == m) & (res.run_tag == s) & (res.test == t)].auc
                if len(v):
                    mat[i, j] = v.mean()
        ax.imshow(mat, cmap="Blues", vmin=np.nanmin(mat) - .05 if np.isfinite(mat).any() else 0, vmax=1)
        for i in range(2):
            for j in range(2):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, idn(mat[i, j]), ha="center", va="center", fontsize=11,
                            color="white" if mat[i, j] > np.nanmean(mat) else "#1b2430")
        ax.set_xticks([0, 1], ["Uji AP", "Uji PA"])
        ax.set_yticks([0, 1], ["Latih AP", "Latih PA"])
        ax.set_title(NAME[m], fontsize=10)
    fig.suptitle("AUC rerata 3 seed (set matched)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def calibration(runs: Path, splits: Path):
    """Sensitivitas/spesifisitas pada ambang global vs ambang khusus posisi (dari validasi per posisi)."""
    val = pd.read_csv(splits / "S1_all" / "val.csv")[["patientId", "view"]]
    rows = []
    for m in MODELS:
        for d in sorted((runs / "S1_all" / m).glob("seed*")):
            vp = d / "val_preds.csv"
            if not vp.exists():
                continue
            v = pd.read_csv(vp).merge(val, on="patientId")
            thr_g = threshold_at_specificity(v.label, v.prob, 0.90)
            for view in ("AP", "PA"):
                f = d / f"preds_{view}_natural.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f)
                sub = v[v.view == view]
                thr_d = threshold_at_specificity(sub.label, sub.prob, 0.90)
                g = binary_metrics(t.label, t.prob, thr_g)
                dd = binary_metrics(t.label, t.prob, thr_d)
                rows.append(dict(model=m, seed=d.name, view=view, sens_g=g["sensitivity"], spec_g=g["specificity"],
                                 sens_d=dd["sensitivity"], spec_d=dd["specificity"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--platform", default="GPU (CUDA)")
    ap.add_argument("--week", default="13")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="laporan/08_Progress 5.docx")
    a = ap.parse_args()

    res = pd.read_csv(a.results).drop_duplicates(["run_dir", "test"], keep="last")
    res = res[res.model.isin(MODELS)]
    if res[res.run_tag.isin(["S2_AP_matched", "S3_PA_matched"])].empty:
        sys.exit("Hasil S2/S3 matched belum ada. Jalankan grup 'p5a' dulu.")
    shift = pd.read_csv(Path(a.tables) / "tabel_domain_shift.csv") if (Path(a.tables) / "tabel_domain_shift.csv").exists() else pd.DataFrame()
    paired = pd.read_csv(Path(a.tables) / "tabel_uji_berpasangan.csv") if (Path(a.tables) / "tabel_uji_berpasangan.csv").exists() else pd.DataFrame()
    figdir = Path(a.out).parent / "gambar_progress5"
    figdir.mkdir(parents=True, exist_ok=True)
    heatmaps(res, figdir / "heatmap_domain.png")
    cal = calibration(Path(a.runs), Path(a.splits))

    def A(m, s, t, c="auc"):
        return res[(res.model == m) & (res.run_tag == s) & (res.test == t)][c]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    para(doc, "LAPORAN PROGRESS KE-5", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
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

    # ringkasan dengan angka gap utama
    def gap(m, test, target, source):
        return A(m, target, test).mean() - A(m, source, test).mean()
    g_pa = {m: gap(m, "PA_matched", "S3_PA_matched", "S2_AP_matched") for m in MODELS}
    g_ap = {m: gap(m, "AP_matched", "S2_AP_matched", "S3_PA_matched") for m in MODELS}
    has_b = not res[res.run_tag.str.contains("_clahe|_posw|_lr", regex=True)].empty
    heading(doc, "1. Ringkasan Progress")
    para(doc, (f"Pada minggu ke-{a.week} skenario inti dijalankan: model dilatih hanya pada satu posisi (S2: AP, S3: PA) "
               f"dengan jumlah data, prevalensi, dan komposisi kelas yang disamakan (matched), lalu diuji pada kedua posisi. "
               f"Pada set uji PA, model yang dilatih pada AP kehilangan AUC sebesar {idn(g_pa[BASE])} (EfficientNet-B0) dan "
               f"{idn(g_pa[MAIN])} (Swin-T) dibanding model yang dilatih pada PA; pada set uji AP selisihnya {idn(g_ap[BASE])} "
               f"dan {idn(g_ap[MAIN])}. "
               + ("Selain itu dijalankan versi natural, ablasi CLAHE dan bobot kelas, tuning learning rate, serta "
                  "kalibrasi ambang per posisi." if has_b else "Versi natural, ablasi, dan tuning dijadwalkan pada run berikutnya.")))
    caption(doc, f"Tabel 1. Status target minggu ke-{a.week}")
    table(doc, [["No", "Target Minggu Ini (sesuai timeline)", "Status", "Keterangan"],
                ["1", "Skenario eksperimen sesuai tujuan topik (S2/S3 matched)", "Selesai", "2 model × 2 skenario × 3 seed"],
                ["2", "Tuning hyperparameter", "Selesai" if has_b else "Belum", "Learning rate 1e-4 / 3e-4 / 1e-3 (EfficientNet-B0)"],
                ["3", "Ablasi", "Selesai" if has_b else "Belum", "CLAHE, bobot kelas, kalibrasi ambang per posisi"]],
          [1.0, 7.0, 2.2, 5.8])

    heading(doc, "2. Uraian Pekerjaan")
    heading(doc, "2.1 Rancangan Skenario", 2)
    para(doc, ("Semua skenario memakai set uji tetap yang sama. Pada versi matched, setiap kelas (Lung Opacity, Not Normal, "
               "Normal) diambil dalam jumlah yang sama dari AP dan PA (6.326 citra latih per posisi), sehingga perbedaan "
               "performa tidak dapat dijelaskan oleh perbedaan prevalensi atau ukuran data. Dua ukuran dipakai: "
               "**drop** = AUC uji posisi sendiri − AUC uji posisi lain (model yang sama), dan **gap** = AUC model "
               "yang dilatih pada posisi target − AUC model yang dilatih pada posisi lain, pada set uji yang sama. Gap "
               "menjadi ukuran utama karena tidak tercampur perbedaan tingkat kesulitan citra AP dan PA."))
    heading(doc, "2.2 Ablasi, Tuning, dan Kalibrasi", 2)
    para(doc, ("Ablasi dilakukan pada EfficientNet-B0 skenario S1: (a) CLAHE untuk meningkatkan kontras lokal, "
               "(b) bobot kelas positif pada loss untuk mengatasi ketidakseimbangan. Tuning learning rate menguji 1e-4 dan "
               "1e-3 terhadap konfigurasi standar 3e-4. Kalibrasi ambang: selain ambang global (spesifisitas 90% pada seluruh "
               "data validasi), dihitung ambang khusus per posisi dari subset validasi AP atau PA, lalu diterapkan pada set uji "
               "posisi yang sesuai. Analisis ini tidak memerlukan pelatihan ulang."))

    heading(doc, "3. Hasil")
    caption(doc, "Tabel 2. AUC seluruh skenario (rerata ± SD; natural: 1 seed)")
    rows = [["Model", "Skenario latih"] + [lab for _, lab in TESTS]]
    for m in MODELS:
        for s, slab in SCEN:
            if res[(res.model == m) & (res.run_tag == s)].empty:
                continue
            rows.append([NAME[m], slab] + [ms(A(m, s, t)) for t, _ in TESTS])
    table(doc, rows, [2.6, 3.4, 2.0, 2.0, 2.0, 2.0, 2.0], size=8)
    figure(doc, figdir / "heatmap_domain.png", 14.5,
           "Gambar 1. AUC lintas posisi: baris = posisi data latih, kolom = posisi data uji")

    if not shift.empty:
        caption(doc, "Tabel 3. Ukuran pergeseran posisi (rerata ± SD antar seed)")
        rows = [["Versi", "Model", "Ukuran", "Nilai"]]
        for _, r in shift.iterrows():
            rows.append([r.varian, NAME.get(r.model, r.model), r.ukuran, r.nilai.replace(".", ",")])
        table(doc, rows, [2.0, 3.0, 7.0, 4.0], size=9)
    if not paired.empty:
        pp = paired[paired.perbandingan.str.startswith("latih AP - latih PA")]
        if len(pp):
            caption(doc, "Tabel 4. Uji bootstrap berpasangan: latih AP − latih PA pada set uji yang sama")
            rows = [["Model", "Versi", "Set uji", "ΔAUC", "CI 95%", "p"]]
            for _, r in pp.iterrows():
                mname = r.perbandingan.split("(")[-1].rstrip(")")
                rows.append([NAME.get(mname, mname), r.skenario, r.uji, idn(r.dAUC),
                             f"[{idn(r.ci_lo)}; {idn(r.ci_hi)}]", idn(r.p)])
            table(doc, rows, [3.0, 2.0, 2.8, 2.0, 3.6, 1.6], size=9)

    abl = [("S1_all", "Standar (lr 3e-4)"), ("S1_all_clahe", "+ CLAHE"), ("S1_all_posw", "+ bobot kelas"),
           ("S1_all_lr1e-4", "lr 1e-4"), ("S1_all_lr1e-3", "lr 1e-3")]
    if has_b:
        caption(doc, "Tabel 5. Ablasi dan tuning EfficientNet-B0 pada S1 (seed 0)")
        rows = [["Konfigurasi", "AUC validasi*", "AUC uji semua", "AUC AP", "AUC PA", "Sens. PA", "AUC Kermany"]]
        for tag, lab in abl:
            x = res[(res.model == BASE) & (res.run_tag == tag) & (res.seed == 0)]
            if x.empty:
                continue
            hf = Path(a.runs) / tag / BASE / "seed0" / "history.csv"
            vauc = idn(pd.read_csv(hf).val_auc.max()) if hf.exists() else "–"

            def g(t, c="auc"):
                v = x[x.test == t][c]
                return idn(v.iloc[0]) if len(v) else "–"
            rows.append([lab, vauc, g("all"), g("AP_natural"), g("PA_natural"), g("PA_natural", "sensitivity"), g("kermany")])
        table(doc, rows, [3.2, 2.2, 2.2, 2.0, 2.0, 2.0, 2.4], size=9)
        para(doc, ("*Pemilihan konfigurasi hanya berdasarkan AUC validasi. Angka uji ditampilkan sebagai informasi, "
                   "tidak dipakai untuk memilih, agar tidak terjadi kebocoran informasi dari set uji."), size=10)
    if not cal.empty:
        caption(doc, "Tabel 6. Kalibrasi ambang: ambang global vs ambang khusus posisi (S1, rerata 3 seed)")
        rows = [["Model", "Set uji", "Sens. global", "Spes. global", "Sens. per posisi", "Spes. per posisi"]]
        for (m, v), g in cal.groupby(["model", "view"]):
            rows.append([NAME[m], f"{v} natural", ms(g.sens_g), ms(g.spec_g), ms(g.sens_d), ms(g.spec_d)])
        table(doc, rows, [3.0, 2.4, 2.6, 2.6, 2.7, 2.7], size=9)

    heading(doc, "4. Analisis dan Pembahasan")
    for m in MODELS:
        d_ap = A(m, "S2_AP_matched", "AP_matched").mean() - A(m, "S2_AP_matched", "PA_matched").mean()
        para(doc, (f"**{NAME[m]}.** Model latih AP: AUC uji AP {idn(A(m, 'S2_AP_matched', 'AP_matched').mean())} vs uji PA "
                   f"{idn(A(m, 'S2_AP_matched', 'PA_matched').mean())} (drop {idn(d_ap)}). Pada set uji PA, gap model latih-PA "
                   f"terhadap latih-AP sebesar {idn(g_pa[m])}; pada set uji AP gap latih-AP terhadap latih-PA {idn(g_ap[m])}. "
                   + ("Pergeseran posisi menurunkan performa secara nyata."
                      if max(g_pa[m], g_ap[m]) > 0.02 else
                      "Pergeseran posisi hanya berdampak kecil setelah faktor pengganggu dikendalikan.")))
    if all(np.isfinite(list(g_pa.values()))):
        more = BASE if (g_pa[BASE] + g_ap[BASE]) > (g_pa[MAIN] + g_ap[MAIN]) else MAIN
        para(doc, (f"**CNN vs transformer.** Total gap {NAME[BASE]} {idn(g_pa[BASE] + g_ap[BASE])} dan {NAME[MAIN]} "
                   f"{idn(g_pa[MAIN] + g_ap[MAIN])}; model yang lebih terpengaruh pergeseran posisi adalah {NAME[more]}. "
                   "Kesimpulan bermakna atau tidaknya merujuk pada Tabel 4."))
    if not cal.empty:
        pa = cal[cal.view == "PA"]
        para(doc, (f"**Kalibrasi ambang.** Dengan ambang global, sensitivitas pada citra PA hanya {ms(pa.sens_g)}; dengan ambang "
                   f"khusus PA menjadi {ms(pa.sens_d)} pada spesifisitas {ms(pa.spec_d)}. Artinya sebagian besar masalah "
                   "sensitivitas pada PA berasal dari ambang keputusan, bukan dari kemampuan model mengurutkan kasus; "
                   "kalibrasi ambang per lokasi/perangkat menjadi rekomendasi penting sebelum model dipakai."))
    if has_b:
        para(doc, ("**Ablasi dan tuning.** Lihat Tabel 5: konfigurasi dengan **AUC validasi** tertinggi menjadi kandidat "
                   "konfigurasi final (set uji tidak dipakai untuk memilih). [CEK] Tuliskan konfigurasi terbaik dan apakah "
                   "selisihnya berarti (selisih < 0,005 sebaiknya dianggap setara karena hanya 1 seed)."))

    heading(doc, "5. Kendala dan Solusi")
    table(doc, [["No", "Kendala", "Dampak", "Solusi / Rencana Penanganan"],
                ["1", "Jumlah run banyak (20 run) dengan GPU gratis terbatas", "Pelatihan memakan beberapa sesi",
                 "Dibagi dua tahap (p5a, p5b); hasil disimpan di Drive dan run selesai dilewati"],
                ["2", "Versi natural, ablasi, dan tuning hanya 1 seed", "Ketidakpastian lebih besar",
                 "Disebutkan sebagai keterbatasan; keputusan utama memakai skenario matched 3 seed"],
                ["3", "Keparahan lesi AP dan PA tetap berbeda walau matched", "Gap tidak murni efek posisi",
                 "Dicatat sebagai keterbatasan; analisis per luas lesi pada Progress 6"]],
          [1.0, 5.0, 4.5, 5.5], size=10)
    heading(doc, "6. Kontribusi Anggota")
    table(doc, [["Nama", "Tugas yang Dikerjakan", "Bukti (commit/file)", "Kontribusi (%)"],
                [TEAM[0][0], "Koordinasi dan menjalankan eksperimen di Colab", "notebooks/colab_progress5.ipynb", "20"],
                [TEAM[1][0], "Skenario S2/S3 matched dan natural", "runs/S2_*, runs/S3_*", "20"],
                [TEAM[2][0], "Uji berpasangan dan kalibrasi ambang", "results/tables/", "20"],
                [TEAM[3][0], "Ablasi CLAHE, bobot kelas, tuning LR", "runs/S1_all_*", "20"],
                [TEAM[4][0], "Penyusunan laporan Progress 5", "Laporan Progress 5", "20"]],
          [4.5, 5.5, 4.2, 1.8], size=10)
    para(doc, ("**Penggunaan alat AI generatif:** Claude (Anthropic) membantu menyusun kode eksperimen dan pembuat laporan "
               "otomatis; semua angka berasal dari keluaran program. [CEK] Sesuaikan pembagian tugas dan kontribusi."))
    heading(doc, "7. Rencana Minggu Berikutnya")
    table(doc, [["No", "Kegiatan", "PIC", "Target Luaran"],
                ["1", "Grad-CAM vs bounding box (pointing game, energy-in-box, border energy)", TEAM[2][0], "Gambar & tabel Grad-CAM"],
                ["2", "Analisis kesalahan: contoh FP/FN, per luas lesi, per subkelompok", TEAM[1][0], "Tabel & contoh kesalahan"],
                ["3", "Confusion matrix final dan efisiensi (latensi, FPS, ukuran)", TEAM[3][0], "Tabel hasil final"],
                ["4", "Menghubungkan model terbaik ke aplikasi mobile", TEAM[0][0], "Demo berjalan"],
                ["5", "Menyusun laporan Progress 6", TEAM[4][0], "Laporan Progress 6"]],
          [1.0, 7.5, 4.0, 3.5], size=10)
    heading(doc, "Referensi")
    for ref in [
        "[1] J. R. Zech et al., “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs,” PLoS Med., vol. 15, no. 11, e1002683, 2018.",
        "[2] J. P. Cohen et al., “On the limits of cross-domain generalization in automated X-ray prediction,” in Proc. MIDL, 2020, pp. 136–155.",
        "[3] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in Graphics Gems IV, Academic Press, 1994, pp. 474–485.",
        "[4] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
    ]:
        p = para(doc, ref, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)
        p.paragraph_format.left_indent = Cm(1)
        p.paragraph_format.first_line_indent = Cm(-1)
    heading(doc, "Lampiran")
    para(doc, "A. Repositori: https://github.com/monicabortang/Project_Paru_CXR", align=WD_ALIGN_PARAGRAPH.LEFT)
    para(doc, "B. Berkas hasil: runs/S2_*, runs/S3_*, runs/S1_all_*, results/all_results.csv, results/tables/.",
         align=WD_ALIGN_PARAGRAPH.LEFT)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"Laporan tersimpan: {out}")
    print(f"  gap uji PA: EffNet {idn(g_pa[BASE])}, Swin {idn(g_pa[MAIN])} | ablasi/tuning: {'ada' if has_b else 'belum'}")


if __name__ == "__main__":
    main()
