@echo off
REM ============================================================
REM  Progress 2 di laptop Windows (CPU cukup): konversi, EDA, split
REM  Jalankan dari Anaconda Prompt, di dalam folder Project_Paru_CXR:
REM      conda activate paru
REM      scripts\progress2_windows.bat
REM  Ubah RSNA_DIR di bawah ke folder hasil ekstrak dataset Anda.
REM ============================================================
REM Default: dataset RSNA diekstrak langsung di folder project (satu tingkat di atas scripts\)
set RSNA_DIR=%~dp0..
set DATA=data
set SPLITS=splits

if not exist "%RSNA_DIR%\stage_2_train_labels.csv" (
    echo [ERROR] stage_2_train_labels.csv tidak ditemukan di %RSNA_DIR%
    echo Ubah baris "set RSNA_DIR=" di file ini.
    exit /b 1
)

echo === 1/3 Konversi DICOM ke PNG + metadata (30-60 menit di CPU) ===
if not exist "%DATA%\metadata.csv" (
    python -m src.prepare_data --rsna_dir "%RSNA_DIR%" --out_dir %DATA% --size 512 --workers 4
    if errorlevel 1 exit /b 1
) else (
    echo metadata.csv sudah ada, dilewati.
)

echo === 2/3 EDA ===
python -m src.eda --meta %DATA%\metadata.csv --img_root %DATA% --out results\eda
if errorlevel 1 exit /b 1

echo === 3/3 Pembagian data (seed 42) ===
python -m src.make_splits --meta %DATA%\metadata.csv --out %SPLITS% --seed 42
if errorlevel 1 exit /b 1

echo.
echo Selesai. Hasil: results\eda\  dan  splits\summary.csv
