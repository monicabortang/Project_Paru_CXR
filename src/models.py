"""Definisi model (timm) + hyperparameter default + layer target Grad-CAM.

Pilihan final proyek:
  baseline    : efficientnet_b0                (CNN, 5,3 jt parameter)
  model utama : swin_tiny_patch4_window7_224   (Transformer hierarkis, 28 jt parameter)
Cadangan (bila waktu/GPU memungkinkan): resnet50, vit_base_patch16_224, convnext_tiny.
"""
from __future__ import annotations

import timm
import torch.nn as nn

MODEL_DEFAULTS = {
    # nama timm                       lr      batch  img
    "efficientnet_b0":              dict(lr=3e-4, batch_size=64, img_size=224),
    "swin_tiny_patch4_window7_224": dict(lr=1e-4, batch_size=48, img_size=224),
    "resnet50":                     dict(lr=3e-4, batch_size=48, img_size=224),
    "vit_base_patch16_224":         dict(lr=5e-5, batch_size=32, img_size=224),
    "convnext_tiny":                dict(lr=1e-4, batch_size=48, img_size=224),
}

ALIASES = {"effnet": "efficientnet_b0", "swin": "swin_tiny_patch4_window7_224",
           "vit": "vit_base_patch16_224"}


def resolve_name(name: str) -> str:
    return ALIASES.get(name, name)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Klasifikasi biner -> 1 logit keluaran."""
    return timm.create_model(resolve_name(name), pretrained=pretrained, num_classes=1)


def gradcam_target(model: nn.Module, name: str):
    """Kembalikan (layer target, fungsi reshape aktivasi -> B,C,H,W)."""
    name = resolve_name(name)
    if name.startswith("efficientnet"):
        return model.bn2, None
    if name.startswith("resnet"):
        return model.layer4, None
    if name.startswith("convnext"):
        return model.stages[-1], None
    if name.startswith("swin"):
        # timm >= 0.9: keluaran stage Swin berformat B,H,W,C
        def reshape(t):
            return t.permute(0, 3, 1, 2) if t.dim() == 4 else t
        return model.layers[-1], reshape
    if name.startswith("vit"):
        n_prefix = getattr(model, "num_prefix_tokens", 1)

        def reshape(t):
            t = t[:, n_prefix:, :]
            s = int(round(t.shape[1] ** 0.5))
            return t.reshape(t.shape[0], s, s, t.shape[2]).permute(0, 3, 1, 2)
        return model.blocks[-1].norm1, reshape
    raise ValueError(f"Target Grad-CAM belum didefinisikan untuk {name}")
