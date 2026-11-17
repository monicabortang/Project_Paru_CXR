"""Server ParuScan: API inferensi + Web App mobile (PWA).

Menjalankan (dari folder project):
  python -m mobile_app.server --run runs/S1_all/swin_tiny_patch4_window7_224/seed0
  python -m mobile_app.server --demo          # uji tampilan sebelum model dilatih

Lalu buka di HP (satu jaringan Wi-Fi dengan laptop): http://<IP-laptop>:8000
Alamat IP ditampilkan saat server mulai.

Endpoint:
  GET  /             aplikasi web mobile
  GET  /api/info     info model (nama, skenario latih, ambang, mode demo)
  POST /api/predict  form-data 'image' (PNG/JPG/DICOM) -> JSON hasil + Grad-CAM
"""
from __future__ import annotations

import argparse
import socket
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

STATIC = Path(__file__).parent / "static"
MAX_MB = 20


def create_app(predictor) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

    @app.after_request
    def headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Cache-Control"] = "no-store" if request.path.startswith("/api") else "no-cache"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        return send_from_directory(STATIC, path)

    @app.get("/api/info")
    def info():
        return jsonify(predictor.info())

    @app.post("/api/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="Tidak ada file gambar yang dikirim."), 400
        data = f.read()
        if not data:
            return jsonify(error="File kosong."), 400
        try:
            return jsonify(predictor.predict(data, f.filename))
        except ValueError as e:
            return jsonify(error=str(e)), 422
        except Exception as e:  # noqa: BLE001
            app.logger.exception("Gagal memproses gambar")
            return jsonify(error=f"Gagal memproses gambar: {type(e).__name__}"), 500

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify(error=f"Ukuran file melebihi {MAX_MB} MB."), 413

    return app


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="folder run berisi best.pt")
    ap.add_argument("--demo", action="store_true", help="model belum dilatih (uji tampilan)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if not a.demo and not a.run:
        ap.error("isi --run <folder run> atau gunakan --demo")

    from .predictor import Predictor

    print("Memuat model ...", flush=True)
    pred = Predictor(a.run, demo=a.demo)
    print(f"Model: {pred.info()}")
    app = create_app(pred)
    print("\n=== ParuScan siap ===")
    print(f"  Laptop : http://localhost:{a.port}")
    for ip in lan_ips():
        print(f"  HP     : http://{ip}:{a.port}   (HP harus satu Wi-Fi dengan laptop)")
    print("Tekan Ctrl+C untuk berhenti.\n", flush=True)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
