# Deploy ParuScan ke Hugging Face Spaces (SDK: Docker) agar bisa diakses HP lewat HTTPS.
# Struktur yang di-upload ke Space:
#   Dockerfile (file ini, taruh di root Space)
#   requirements-mobile.txt
#   src/            (dari repo)
#   mobile_app/     (dari repo)
#   model/best.pt   (salin dari runs/<skenario>/<model>/seed0/best.pt)
FROM python:3.11-slim
WORKDIR /app
COPY requirements-mobile.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-mobile.txt
COPY src ./src
COPY mobile_app ./mobile_app
COPY model ./model
EXPOSE 7860
CMD ["python", "-m", "mobile_app.server", "--run", "model", "--port", "7860"]
