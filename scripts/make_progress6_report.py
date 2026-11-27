"""Progress 6 - Evaluasi menyeluruh: Grad-CAM vs bounding box, analisis kesalahan,
confusion matrix, efisiensi (latensi/FPS/ukuran), tabel hasil final -> laporan .docx otomatis.

Di Colab (GPU): menjalankan Grad-CAM dulu bila belum ada, lalu membuat laporan.
  python scripts/make_progress6_report.py --platform "Google Colab (GPU NVIDIA T4)" --img_root /content/data
Tanpa GPU / Grad-CAM sudah ada:
  python scripts/make_progress6_report.py --skip_gradcam
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
                                   caption, figure, heading, idn, ms, para, pct, qn, table)
from src.common import binary_metrics  # noqa: E402

BASE, MAIN = "efficientnet_b0", "swin_tiny_patch4_window7_224"
MODELS = [BASE, MAIN]
NAME = {BASE: "EfficientNet-B0", MAIN: "Swin-T"}


def ensemble(runs: Path, model: str, test: str = "all"):
    files = sorted((runs / "S1_all" / model).glob(f"seed*/preds_{test}.csv"))
    if not files:
        return None
    base = pd.read_csv(files[0])
    probs = [pd.read_csv(f).set_index("patientId").loc[base.patientId, "prob"].values for f in files]
    base["prob"] = np.mean(probs, axis=0)
    return base


def lesion_area(s):
    try:
        b = json.loads(s) if isinstance(s, str) else []
    except ValueError:
        b = []
    return sum(w * h for _, _, w, h in b)


def run_gradcam(runs: Path, img_root: str):
    for m in MODELS:
        rd = runs / "S1_all" / m / "seed0"
        if (rd / "best.pt").exists() and (rd / "preds_all.csv").exists() \
                and not (rd / "gradcam_summary_all.json").exists():
            print(f"Grad-CAM {m} ...", flush=True)
            subprocess.run([sys.executable, "-m", "src.gradcam", "--run", str(rd), "--test_csv",
                            str(rd / "preds_all.csv"), "--img_root", img_root, "--max_images", "500"],
                           check=True, cwd=HERE.parent)


def plot_cm(cms, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(3.6 * len(cms), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (m, mt) in zip(axes, cms.items()):
        cm = np.array([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() / 2 else "#1b2430")
        ax.set_xticks([0, 1], ["Negatif", "Positif"])
        ax.set_yticks([0, 1], ["Negatif", "Positif"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label")
        ax.set_title(NAME[m], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_miss_by_size(tab, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(tab.index))
    w = .36
    for i, (m, c) in enumerate([(BASE, "#2a78d6"), (MAIN, "#eb6834")]):
        if m in tab:
            b = ax.bar(x + (i - .5) * (w + .02), tab[m].values * 100, w, color=c, label=NAME[m], zorder=3)
            for r, v in zip(b, tab[m].values):
                ax.text(r.get_x() + r.get_width() / 2, v * 100 + 1, f"{v * 100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("Kasus positif terlewat (%)")
    ax.yaxis.grid(True, alpha=.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--img_root", default="data")
    ap.add_argument("--skip_gradcam", action="store_true")
    ap.add_argument("--platform", default="GPU (CUDA)")
    ap.add_argument("--week", default="14")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="laporan/08_Progress 6.docx")
    a = ap.parse_args()
    runs = Path(a.runs)
    if not a.skip_gradcam:
        run_gradcam(runs, a.img_root)

    res = pd.read_csv(a.results).drop_duplicates(["run_dir", "test"], keep="last")
    res = res[(res.run_tag == "S1_all") & res.model.isin(MODELS)]
    models = [m for m in MODELS if not res[res.model == m].empty]
    if not models:
        sys.exit("Hasil S1 belum ada.")
    figdir = Path(a.out).parent / "gambar_progress6"
    figdir.mkdir(parents=True, exist_ok=True)

    # ---------- analisis kesalahan pada prediksi rata-rata 3 seed
    ens, cms, thr = {}, {}, {}
    for m in models:
        e = ensemble(runs, m)
        if e is None:
            continue
        thr[m] = res[(res.model == m) & (res.test == "all")].threshold.mean()
        e["pred"] = (e.prob >= thr[m]).astype(int)
        e["area"] = e.boxes.map(lesion_area) / (e.orig_h * e.orig_w) * 100 if "boxes" in e else np.nan
        ens[m] = e
        cms[m] = binary_metrics(e.label, e.prob, thr[m])
    if cms:
        plot_cm(cms, figdir / "confusion.png")
    size_tab = pd.DataFrame()
    if ens:
        any_e = next(iter(ens.values()))
        pos_area = any_e[any_e.label == 1].area
        q1, q2 = np.nanpercentile(pos_area, [33.3, 66.7])
        lab = [f"Kecil (<{q1:.1f}%)".replace(".", ","), f"Sedang ({q1:.1f}–{q2:.1f}%)".replace(".", ","),
               f"Besar (>{q2:.1f}%)".replace(".", ",")]
        for m, e in ens.items():
            p = e[e.label == 1].copy()
            p["bin"] = pd.cut(p.area, [-np.inf, q1, q2, np.inf], labels=lab)
            size_tab[m] = p.groupby("bin", observed=False).pred.apply(lambda s: 1 - s.mean())
        plot_miss_by_size(size_tab, figdir / "miss_by_size.png")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    para(doc, "LAPORAN PROGRESS KE-6", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
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

    def col(m, t, c):
        return res[(res.model == m) & (res.test == t)][c]

    gc = {}
    for m in models:
        f = runs / "S1_all" / m / "seed0" / "gradcam_summary_all.json"
        if f.exists():
            gc[m] = json.loads(f.read_text())

    heading(doc, "1. Ringkasan Progress")
    def val_auc(m):
        v = [pd.read_csv(h).val_auc.max() for h in (runs / "S1_all" / m).glob("seed*/history.csv")]
        return np.mean(v) if v else -1
    best = max(models, key=val_auc)
    para(doc, (f"Pada minggu ke-{a.week} kedua model dievaluasi menyeluruh pada set uji tetap. Berdasarkan AUC validasi "
               f"(bukan set uji, untuk menghindari kebocoran), model terpilih adalah {NAME[best]}; AUC ujinya "
               f"{ms(col(best, 'all', 'auc'))}. Analisis kesalahan pada prediksi rata-rata 3 seed "
               "menunjukkan bagaimana kasus terlewat bergantung pada luas lesi dan posisi pengambilan; Grad-CAM "
               "dibandingkan dengan bounding box radiolog untuk menilai apakah model melihat area paru yang tepat; "
               "efisiensi diukur sebagai latensi, FPS, dan ukuran model."))
    caption(doc, f"Tabel 1. Status target minggu ke-{a.week}")
    table(doc, [["No", "Target Minggu Ini (sesuai timeline)", "Status", "Keterangan"],
                ["1", "Evaluasi menyeluruh dan tabel hasil final", "Selesai", "Tabel 2"],
                ["2", "Analisis kesalahan dan confusion matrix", "Selesai" if cms else "Belum", "Tabel 3–4, Gambar 1–2"],
                ["3", "Visualisasi Grad-CAM", "Selesai" if gc else "Belum", "Tabel 5, Gambar 3–4"],
                ["4", "Pengukuran efisiensi (latensi, FPS, ukuran)", "Selesai", "Tabel 6"]],
          [1.0, 7.0, 2.2, 5.8])

    heading(doc, "2. Uraian Pekerjaan")
    para(doc, ("**Analisis kesalahan.** Prediksi tiga seed dirata-ratakan (ensembel) lalu dibandingkan dengan label pada "
               "ambang validasi (spesifisitas 90%). Kasus positif dikelompokkan menurut luas total bounding box (tertil "
               "kecil/sedang/besar), dan kasus negatif menurut subkelas (Normal vs No Lung Opacity/Not Normal). "
               "**Grad-CAM** [1] dihitung pada 500 citra positif set uji (model seed 0). Kesesuaian dengan bounding box "
               "diukur dengan pointing game (titik terpanas di dalam kotak), energy-in-box (proporsi energi di dalam "
               "kotak), IoU, dan border energy (energi di tepi citra, indikator shortcut seperti penanda teks). "
               "**Efisiensi** diukur di GPU: latensi batch 1, FPS = 1000/latensi, throughput batch 32, dan ukuran file."))

    heading(doc, "3. Hasil")
    caption(doc, "Tabel 2. Hasil final skenario S1 (rerata ± SD, 3 seed)")
    rows = [["Model", "Set uji", "AUC", "Macro-F1", "Sensitivitas", "Spesifisitas"]]
    for m in models:
        for t, lab in [("all", "Semua"), ("AP_natural", "AP"), ("PA_natural", "PA"), ("kermany", "Kermany")]:
            if col(m, t, "auc").empty:
                continue
            rows.append([NAME[m], lab, ms(col(m, t, "auc")), ms(col(m, t, "macro_f1")),
                         ms(col(m, t, "sensitivity")), ms(col(m, t, "specificity"))])
    table(doc, rows, [3.2, 2.2, 2.8, 2.6, 2.6, 2.6], size=9)
    if cms:
        figure(doc, figdir / "confusion.png", 12.5,
               "Gambar 1. Confusion matrix prediksi rata-rata 3 seed pada set uji semua posisi (n = 3.812)")
    if len(size_tab):
        caption(doc, "Tabel 3. Proporsi kasus positif yang terlewat (false negative) menurut luas lesi")
        rows = [["Luas lesi (% citra)"] + [NAME[m] for m in size_tab.columns]]
        for idx, r in size_tab.iterrows():
            rows.append([idx] + [pct(v) for v in r.values])
        table(doc, rows, [5.0, 4.0, 4.0], size=10)
        figure(doc, figdir / "miss_by_size.png", 12.5, "Gambar 2. Kasus positif terlewat menurut luas lesi")
    if ens:
        caption(doc, "Tabel 4. Kesalahan per subkelompok (prediksi rata-rata 3 seed)")
        rows = [["Subkelompok", *[f"{NAME[m]}" for m in ens]]]

        def rate(m, mask, kind):
            e = ens[m][mask(ens[m])]
            if kind == "fpr":
                return pct((e.pred == 1).mean()) if len(e) else "–"
            return pct((e.pred == 0).mean()) if len(e) else "–"
        rows.append(["FP pada Normal", *[rate(m, lambda e: e.class3 == "Normal", "fpr") for m in ens]])
        rows.append(["FP pada Not Normal", *[rate(m, lambda e: e.class3 == "No Lung Opacity / Not Normal", "fpr") for m in ens]])
        for v in ("AP", "PA"):
            rows.append([f"FN pada {v}", *[rate(m, lambda e, v=v: (e.view == v) & (e.label == 1), "fnr") for m in ens]])
        for s_, lab in (("M", "laki-laki"), ("F", "perempuan")):
            rows.append([f"FN pada {lab}", *[rate(m, lambda e, s_=s_: (e.sex == s_) & (e.label == 1), "fnr") for m in ens]])
        for lo, hi, lab in ((0, 40, "<40 th"), (40, 60, "40–59 th"), (60, 200, "≥60 th")):
            rows.append([f"FN usia {lab}", *[rate(m, lambda e, lo=lo, hi=hi: (e.age >= lo) & (e.age < hi) & (e.label == 1), "fnr")
                                            for m in ens]])
        table(doc, rows, [5.0, 4.0, 4.0], size=10)
    if gc:
        caption(doc, "Tabel 5. Kesesuaian Grad-CAM dengan bounding box radiolog (citra positif, seed 0)")
        rows = [["Metrik", *[NAME[m] for m in gc], "Acuan acak"]]
        keys = [("pointing_game", "Pointing game"), ("energy_in_box", "Energy-in-box"), ("iou@0.5", "IoU (CAM ≥ 0,5)"),
                ("border_energy", "Border energy")]
        any_g = next(iter(gc.values()))["all_positive"]
        for k, lab in keys:
            ref = idn(any_g.get("box_area_frac", np.nan)) if k in ("pointing_game", "energy_in_box") else "–"
            rows.append([lab, *[idn(gc[m]["all_positive"].get(k, np.nan)) for m in gc], ref])
        table(doc, rows, [4.6, 3.6, 3.6, 3.6], size=10)
        para(doc, ("Acuan acak = proporsi luas bounding box terhadap citra, yaitu nilai yang diharapkan bila peta perhatian "
                   "tersebar acak. Nilai di atas acuan berarti model memusatkan perhatian pada area lesi."), size=10)
        for i, m in enumerate(gc, 3):
            f = runs / "S1_all" / m / "seed0" / "gradcam_examples_all.png"
            if f.exists():
                figure(doc, f, 13.0, f"Gambar {i}. Contoh Grad-CAM {NAME[m]}: benar positif (TP), terlewat (FN), "
                                     "dan positif palsu (FP); kotak hijau = bounding box radiolog")
    # ---------- analisis sensitivitas kebocoran tingkat pasien (hasil audit kemiripan citra)
    leak_f = Path("splits/audit_leak_suspects.csv")
    if leak_f.exists() and ens:
        from sklearn.metrics import roc_auc_score
        sus = set(pd.read_csv(leak_f).patientId)
        caption(doc, "Tabel 7. Analisis sensitivitas kebocoran pasien: AUC dengan dan tanpa citra uji yang dicurigai")
        rows = [["Model", "AUC semua (n=3.812)", f"AUC tanpa tersangka (n={3812 - len(sus)})", "Selisih"]]
        for m, e in ens.items():
            a1 = roc_auc_score(e.label, e.prob)
            k = e[~e.patientId.isin(sus)]
            a2 = roc_auc_score(k.label, k.prob)
            rows.append([NAME[m], idn(a1), idn(a2), idn(a2 - a1)])
        table(doc, rows, [3.4, 3.8, 4.8, 2.4], size=10)
        para(doc, (f"Audit kemiripan citra (korelasi piksel 32×32 terhadap seluruh data latih/validasi, dikonfirmasi dengan "
                   f"kecocokan jenis kelamin, posisi, dan usia) menemukan {len(sus)} citra uji yang kemungkinan berasal dari "
                   "pasien yang juga ada di data latih/validasi, karena patientId RSNA adalah ID per citra. Bila AUC tanpa "
                   "citra tersebut hampir sama, kebocoran ini tidak memengaruhi kesimpulan."), size=11)

    eff = {}
    for m in models:
        f = runs / "S1_all" / m / "seed0" / "metrics.json"
        if f.exists():
            eff[m] = json.loads(f.read_text()).get("efficiency", {})
    if eff:
        caption(doc, "Tabel 6. Efisiensi model (GPU, seed 0)")
        rows = [["Model", "Parameter", "Ukuran", "Latensi (batch 1)", "FPS", "Throughput (batch 32)"]]
        for m, e in eff.items():
            lat = e.get("latency_ms_bs1", np.nan)
            rows.append([NAME[m], f"{idn(e.get('params_M', np.nan), 2)} jt", f"{idn(e.get('size_MB', np.nan), 1)} MB",
                         f"{idn(lat, 1)} ms", idn(1000 / lat, 0) if lat else "–",
                         f"{idn(e.get('throughput_img_s_bs32', np.nan), 0)} citra/s"])
        table(doc, rows, [3.0, 2.2, 2.2, 3.0, 1.8, 3.8], size=10)

    heading(doc, "4. Analisis dan Interpretasi")
    if len(size_tab):
        for m in size_tab.columns:
            v = size_tab[m].values
            para(doc, (f"**Luas lesi ({NAME[m]}).** Kasus terlewat {pct(v[0])} pada lesi kecil, {pct(v[1])} sedang, dan "
                       f"{pct(v[2])} besar. " + ("Model paling sering melewatkan lesi kecil, sejalan dengan rendahnya "
                                                  "sensitivitas pada citra PA yang lesinya lebih kecil (Progress 2)."
                                                  if v[0] > v[2] else "Pola kesalahan tidak bergantung kuat pada luas lesi.")))
    if gc:
        for m, g in gc.items():
            x = g["all_positive"]
            para(doc, (f"**Grad-CAM ({NAME[m]}).** Pointing game {idn(x.get('pointing_game'))} dan energy-in-box "
                       f"{idn(x.get('energy_in_box'))} dibanding acuan acak {idn(x.get('box_area_frac'))}; border energy "
                       f"{idn(x.get('border_energy'))}. "
                       + ("Model cenderung melihat area lesi." if x.get("energy_in_box", 0) > 1.5 * x.get("box_area_frac", 1)
                          else "Perhatian model belum terpusat pada lesi; sebagian prediksi mungkin memanfaatkan petunjuk lain.")
                       + (" Energi di tepi citra cukup tinggi, perlu diwaspadai sebagai shortcut (penanda teks/alat)."
                          if x.get("border_energy", 0) > 0.25 else "")))
    if eff and len(eff) == 2:
        para(doc, (f"**Efisiensi.** EfficientNet-B0 sekitar {idn(eff[MAIN]['latency_ms_bs1'] / eff[BASE]['latency_ms_bs1'], 1)}× "
                   "lebih cepat per citra dan jauh lebih kecil; keduanya tetap layak untuk skrining (di bawah 50 ms per "
                   "citra di GPU). Untuk aplikasi mobile yang berjalan di server CPU, model yang lebih ringan lebih praktis. "
                   "[CEK] Tuliskan model yang dipilih untuk demo beserta alasannya."))

    heading(doc, "5. Kendala dan Solusi")
    table(doc, [["No", "Kendala", "Dampak", "Solusi / Rencana Penanganan"],
                ["1", "Grad-CAM hanya pendekatan kasar (resolusi 7×7)", "Lokalisasi kurang presisi",
                 "Dipakai sebagai indikator, bukan alat lokalisasi; dilengkapi metrik kuantitatif"],
                ["2", "Ambang global kurang cocok untuk PA dan data eksternal", "Sensitivitas rendah",
                 "Rekomendasi kalibrasi ambang per lokasi (Progress 5)"]],
          [1.0, 5.0, 4.5, 5.5], size=10)
    heading(doc, "6. Kontribusi Anggota")
    table(doc, [["Nama", "Tugas yang Dikerjakan", "Bukti (commit/file)", "Kontribusi (%)"],
                [TEAM[0][0], "Koordinasi dan menjalankan evaluasi", "notebooks/colab_progress6.ipynb", "20"],
                [TEAM[1][0], "Grad-CAM dan metrik lokalisasi", "runs/S1_all/*/seed0/gradcam_*", "20"],
                [TEAM[2][0], "Analisis kesalahan per subkelompok", "Tabel 3–4", "20"],
                [TEAM[3][0], "Pengukuran efisiensi", "metrics.json", "20"],
                [TEAM[4][0], "Penyusunan laporan Progress 6", "Laporan Progress 6", "20"]],
          [4.5, 5.5, 4.2, 1.8], size=10)
    para(doc, ("**Penggunaan alat AI generatif:** Claude (Anthropic) membantu menyusun kode analisis dan pembuat laporan "
               "otomatis; semua angka berasal dari keluaran program. [CEK] Sesuaikan pembagian tugas dan kontribusi."))
    heading(doc, "7. Rencana Minggu Berikutnya")
    table(doc, [["No", "Kegiatan", "PIC", "Target Luaran"],
                ["1", "Menghubungkan model terpilih ke aplikasi mobile ParuScan", TEAM[0][0], "Tangkapan layar demo"],
                ["2", "Merapikan kode dan README repositori", TEAM[3][0], "Repositori siap dinilai"],
                ["3", "Menyusun draf laporan akhir lengkap", TEAM[4][0], "Draf laporan akhir"],
                ["4", "Menyiapkan slide presentasi", TEAM[1][0], "Draf slide"],
                ["5", "Merangkum temuan dan keterbatasan", TEAM[2][0], "Bab hasil & pembahasan"]],
          [1.0, 7.5, 4.0, 3.5], size=10)
    heading(doc, "Referensi")
    for ref in [
        "[1] R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE ICCV, 2017, pp. 618–626.",
        "[2] J. Zhang et al., “Top-down neural attention by excitation backprop,” Int. J. Comput. Vis., vol. 126, pp. 1084–1102, 2018.",
        "[3] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 detection selects shortcuts over signal,” Nat. Mach. Intell., vol. 3, pp. 610–619, 2021.",
    ]:
        p = para(doc, ref, size=11, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)
        p.paragraph_format.left_indent = Cm(1)
        p.paragraph_format.first_line_indent = Cm(-1)
    heading(doc, "Lampiran")
    para(doc, "A. Repositori: https://github.com/monicabortang/Project_Paru_CXR", align=WD_ALIGN_PARAGRAPH.LEFT)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"Laporan tersimpan: {out}")


if __name__ == "__main__":
    main()
