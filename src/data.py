"""Dataset PyTorch dan transformasi citra."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .common import load_gray

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_three_channels(x):
    """Grayscale 1xHxW -> 3xHxW. Fungsi level-modul (bukan lambda) agar bisa
    di-pickle oleh DataLoader multi-worker di Windows."""
    return x.expand(3, -1, -1)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if h == w:
        return img
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def build_transform(img_size: int, train: bool, hflip: bool = False):
    ops = [T.ToImage(), T.Resize((img_size, img_size), antialias=True)]
    if train:
        ops += [
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if hflip:  # default mati: membalik CXR menukar posisi jantung
            ops.append(T.RandomHorizontalFlip())
    ops += [
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(to_three_channels),  # grayscale -> 3 kanal
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


def preprocess_array(img: np.ndarray, img_size: int, clahe: bool = False) -> torch.Tensor:
    """Untuk inferensi tunggal (demo/Grad-CAM): uint8 HxW -> tensor 1x3xSxS."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    img = pad_to_square(img)
    return build_transform(img_size, train=False)(img[..., None]).unsqueeze(0)


class CXRDataset(Dataset):
    def __init__(self, csv: str | Path | pd.DataFrame, img_root: str | Path,
                 img_size: int = 224, train: bool = False, clahe: bool = False,
                 hflip: bool = False):
        self.df = csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv)
        self.df = self.df.reset_index(drop=True)
        self.root = Path(img_root)
        self.clahe = clahe
        self.tf = build_transform(img_size, train, hflip)

    def __len__(self):
        return len(self.df)

    def _path(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = pad_to_square(load_gray(self._path(r.img_path), self.clahe))
        x = self.tf(img[..., None])  # HxWx1 -> 1xHxW
        return x, torch.tensor(float(r.label)), i
