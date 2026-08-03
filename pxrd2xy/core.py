"""Core image utilities: loading, background/ink masks, run-length helpers."""
from __future__ import annotations

import numpy as np
import cv2


def load_rgb(path: str) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB, compositing any alpha over white."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha))
        img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def upscale_if_small(rgb: np.ndarray, min_width: int = 900) -> tuple[np.ndarray, float]:
    """Upscale small figures so line width / ticks are resolvable. Returns (img, scale)."""
    h, w = rgb.shape[:2]
    if w >= min_width:
        return rgb, 1.0
    scale = float(np.ceil(min_width / w))
    # nearest-neighbour: keeps the original colour palette exactly, so anti-aliasing
    # ramps are not turned into new intermediate colours that split colour clusters
    out = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return out, scale


def background_color(rgb: np.ndarray) -> np.ndarray:
    """Most common (coarsely quantized) colour, taken as the plot background."""
    q = (rgb // 16).reshape(-1, 3)
    keys = q[:, 0].astype(np.int32) * 4096 + q[:, 1].astype(np.int32) * 64 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[counts.argmax()]
    bg_q = np.array([k // 4096, (k // 64) % 64, k % 64])
    lo, hi = bg_q * 16, bg_q * 16 + 15
    sel = np.all((rgb.reshape(-1, 3) >= lo) & (rgb.reshape(-1, 3) <= hi), axis=1)
    return rgb.reshape(-1, 3)[sel].mean(axis=0)


def ink_mask(rgb: np.ndarray, bg: np.ndarray, thresh: int = 40) -> np.ndarray:
    """Boolean mask of pixels that differ from the background colour."""
    d = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    return d > thresh


def longest_run(mask: np.ndarray, axis: int) -> np.ndarray:
    """Longest run of True along `axis` for every line of the other axis."""
    m = mask if axis == 1 else mask.T
    n, L = m.shape
    best = np.zeros(n, dtype=np.int32)
    cur = np.zeros(n, dtype=np.int32)
    for j in range(L):
        col = m[:, j]
        cur = np.where(col, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def group_consecutive(idx: np.ndarray, gap: int = 2) -> list[tuple[int, int]]:
    """Group sorted indices into (start, end) inclusive bands, merging gaps <= `gap`."""
    if len(idx) == 0:
        return []
    bands, s, p = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - p <= gap:
            p = v
        else:
            bands.append((s, p))
            s = p = v
    bands.append((s, p))
    return bands


def column_runs(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """For each column, list of (y0, y1) inclusive vertical runs of True."""
    H, W = mask.shape
    out = []
    for x in range(W):
        col = mask[:, x]
        if not col.any():
            out.append([])
            continue
        idx = np.flatnonzero(col)
        out.append(group_consecutive(idx, gap=1))
    return out


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8, N x 3 or HxWx3) -> CIE Lab float32, L in 0..100."""
    arr = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab
