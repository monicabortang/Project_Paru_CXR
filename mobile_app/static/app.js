"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "paruscan_history_v1";
const MAX_UPLOAD_SIDE = 1600; // foto kamera diperkecil sebelum dikirim

let currentFile = null;
let lastResult = null;

// ---------------------------------------------------------------- info server
async function loadInfo() {
  try {
    const r = await fetch("api/info", { cache: "no-store" });
    if (!r.ok) throw new Error();
    const info = await r.json();
    $("demoBanner").hidden = !info.demo;
    $("offlineBanner").hidden = true;
  } catch {
    $("offlineBanner").hidden = false;
  }
}

// ---------------------------------------------------------------- pilih file
function isDicom(file) {
  return /\.dcm$/i.test(file.name) || file.type === "application/dicom";
}

function onFile(file) {
  if (!file) return;
  currentFile = file;
  const img = $("previewImg");
  const chip = $("previewFile");
  $("preview").classList.remove("empty");
  $("preview").querySelector(".placeholder").hidden = true;
  if (isDicom(file)) {
    img.hidden = true;
    chip.hidden = false;
    chip.textContent = `File DICOM: ${file.name}`;
  } else {
    chip.hidden = true;
    img.hidden = false;
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    const url = URL.createObjectURL(file);
    img.dataset.url = url;
    img.src = url;
  }
  $("analyzeBtn").disabled = false;
  $("resultCard").hidden = true;
}

$("cameraInput").addEventListener("change", (e) => onFile(e.target.files[0]));
$("fileInput").addEventListener("change", (e) => onFile(e.target.files[0]));

// Perkecil foto besar (mis. dari kamera 12 MP) agar unggah cepat.
async function shrinkIfNeeded(file) {
  if (isDicom(file) || !/^image\//.test(file.type)) return file;
  try {
    const bmp = await createImageBitmap(file);
    const s = Math.min(1, MAX_UPLOAD_SIDE / Math.max(bmp.width, bmp.height));
    if (s === 1 && file.size < 3 * 1024 * 1024) return file;
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s);
    c.height = Math.round(bmp.height * s);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise((res) => c.toBlob(res, "image/png"));
    return blob ? new File([blob], file.name.replace(/\.\w+$/, "") + ".png", { type: "image/png" }) : file;
  } catch {
    return file;
  }
}

// ---------------------------------------------------------------- analisis
async function analyze() {
  if (!currentFile) return;
  $("loading").hidden = false;
  try {
    const fd = new FormData();
    const f = await shrinkIfNeeded(currentFile);
    fd.append("image", f, f.name);
    const r = await fetch("api/predict", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `Server error (${r.status})`);
    $("offlineBanner").hidden = true;
    showResult(data);
    saveHistory(data);
  } catch (err) {
    const msg = err instanceof TypeError
      ? "Tidak dapat terhubung ke server. Periksa koneksi/jaringan."
      : err.message;
    alertBox(msg);
  } finally {
    $("loading").hidden = true;
  }
}

function alertBox(msg) {
  const b = $("offlineBanner");
  b.textContent = msg;
  b.hidden = false;
  b.scrollIntoView({ behavior: "smooth", block: "center" });
}

const SCENARIO_LABEL = {
  S1_all: "RSNA, semua posisi (AP + PA)",
  S2_AP_matched: "RSNA, posisi AP saja",
  S3_PA_matched: "RSNA, posisi PA saja",
  S2_AP_natural: "RSNA, posisi AP saja",
  S3_PA_natural: "RSNA, posisi PA saja",
};
function pct(x) { return (x * 100).toFixed(1).replace(".", ",") + "%"; }

function showResult(d) {
  lastResult = d;
  const v = $("verdict");
  v.className = "verdict " + (d.positive ? "pos" : "neg");
  v.innerHTML = '<span class="dot"></span>';
  v.append(document.createTextNode(d.verdict));
  $("probValue").textContent = pct(d.probability);
  $("thrValue").textContent = pct(d.threshold);
  const fill = $("barFill");
  fill.className = "fill" + (d.positive ? " pos" : "");
  fill.style.width = "0";
  requestAnimationFrame(() => { fill.style.width = (d.probability * 100).toFixed(1) + "%"; });
  $("barThr").style.left = `calc(${(d.threshold * 100).toFixed(1)}% - 1px)`;
  setTab("cam");
  const w = $("warnings");
  w.innerHTML = "";
  (d.warnings || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; w.append(li); });
  w.hidden = !(d.warnings || []).length;
  $("metaModel").textContent = d.model + (d.demo ? " (demo)" : "");
  $("metaScenario").textContent = SCENARIO_LABEL[d.scenario] || d.scenario;
  $("metaTime").textContent = `${d.elapsed_ms} ms`;
  if (d.original && isDicom(currentFile)) {   // tampilkan pratinjau DICOM dari server
    $("previewImg").src = d.original; $("previewImg").hidden = false; $("previewFile").hidden = true;
  }
  $("resultCard").hidden = false;
  $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setTab(which) {
  const cam = which === "cam";
  $("tabCam").classList.toggle("active", cam);
  $("tabOrig").classList.toggle("active", !cam);
  $("tabCam").setAttribute("aria-selected", cam);
  $("tabOrig").setAttribute("aria-selected", !cam);
  if (lastResult) $("resultImg").src = cam ? lastResult.gradcam : lastResult.original;
  $("camHint").hidden = !cam;
}
$("tabCam").addEventListener("click", () => setTab("cam"));
$("tabOrig").addEventListener("click", () => setTab("orig"));
$("analyzeBtn").addEventListener("click", analyze);
$("againBtn").addEventListener("click", () => {
  $("resultCard").hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
  $("fileInput").click();
});

// ---------------------------------------------------------------- riwayat
function readHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}
function writeHistory(h) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h)); } catch { /* penyimpanan penuh/diblokir */ }
}
function thumb(dataUrl) {
  return new Promise((res) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas");
      c.width = c.height = 96;
      c.getContext("2d").drawImage(im, 0, 0, 96, 96);
      res(c.toDataURL("image/jpeg", 0.7));
    };
    im.onerror = () => res("");
    im.src = dataUrl;
  });
}
async function saveHistory(d) {
  const h = readHistory();
  h.unshift({
    t: new Date().toISOString(), p: d.probability, pos: d.positive,
    v: d.verdict, demo: d.demo, th: await thumb(d.gradcam),
  });
  writeHistory(h.slice(0, 10));
  renderHistory();
}
function renderHistory() {
  const h = readHistory();
  $("historyCard").hidden = h.length === 0;
  const ul = $("historyList");
  ul.innerHTML = "";
  h.forEach((it) => {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.src = it.th || "icons/icon-192.png";
    img.alt = "";
    const tx = document.createElement("div");
    tx.className = "htext";
    const hv = document.createElement("div");
    hv.className = "hv " + (it.pos ? "pos" : "neg");
    hv.textContent = (it.pos ? "Terindikasi" : "Tidak terindikasi") + (it.demo ? " · demo" : "");
    const ht = document.createElement("div");
    ht.className = "ht";
    ht.textContent = new Date(it.t).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
    tx.append(hv, ht);
    const hp = document.createElement("div");
    hp.className = "hp";
    hp.textContent = pct(it.p);
    li.append(img, tx, hp);
    ul.append(li);
  });
}
$("clearHistory").addEventListener("click", () => { writeHistory([]); renderHistory(); });

// ---------------------------------------------------------------- PWA
if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

loadInfo();
renderHistory();
