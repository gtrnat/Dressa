"""Numeric features for the Dressa ML attribute classifier.

This file is the single source of truth for feature extraction. The training notebook
(ml/training/dressa_ml_classifier.ipynb) contains an exact copy of it, so training and
serving compute features the same way. Bump FEATURE_VERSION whenever the output changes.

Pipeline: image -> prepare() -> [deep embedding (pretrained, frozen)] + handcrafted() -> one row of numbers
"""

import numpy as np
from PIL import Image
from scipy import ndimage

FEATURE_VERSION = 1
PREPARED_SIZE = 80  # the Kaggle "small" images are 60x80; serving images are reduced to the same detail level

HUE_BINS = 12
HANDCRAFTED_NAMES: list[str] = (
    ["lab_L_mean", "lab_a_mean", "lab_b_mean", "lab_L_std", "lab_a_std", "lab_b_std"]
    + [f"hue_{i * 30:03d}" for i in range(HUE_BINS)]
    + ["gray_ratio", "dark_ratio", "bright_ratio", "saturation_mean"]
    + ["fg_ratio", "bbox_aspect", "bbox_fill", "centroid_x", "centroid_y", "sym_lr", "sym_tb"]
    + ["width_top", "width_mid", "width_bottom", "height_left", "height_mid", "height_right", "gap_bottom"]
    + ["edge_density", "edge_strength"]
)


def prepare(img: Image.Image) -> Image.Image:
    """RGB on white, padded to a square, resized to PREPARED_SIZE. Used for BOTH the embedding and handcrafted features."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((PREPARED_SIZE, PREPARED_SIZE), Image.Resampling.BILINEAR)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Garment pixels of an image on a white background: near-white regions touching the border are background."""
    near_white = (rgb.min(axis=2) > 235) & (rgb.max(axis=2) - rgb.min(axis=2) < 12)
    labels, _ = ndimage.label(near_white)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, border[border > 0])
    fg = ndimage.binary_fill_holes(~background)
    if fg.mean() < 0.02:  # e.g. a white garment that merged with the background
        fg = np.ones_like(fg, dtype=bool)
    return fg


def handcrafted(prepared: Image.Image) -> np.ndarray:
    """Color, shape and texture features (len == len(HANDCRAFTED_NAMES)) from a prepare()-d image."""
    rgb = np.asarray(prepared.convert("RGB"), dtype=np.uint8)
    fg = foreground_mask(rgb)
    px = rgb[fg].astype(np.float64)

    # --- color (garment pixels only) ---
    lab = _srgb_to_lab(px)
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    d = np.maximum(mx - mn, 1e-9)
    hue = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    chroma = sat >= 0.2
    hist = np.histogram(hue[chroma], bins=HUE_BINS, range=(0, 360), weights=sat[chroma])[0]
    hist = hist / max(hist.sum(), 1e-9)
    color = np.concatenate([
        lab.mean(0), lab.std(0), hist,
        [1 - chroma.mean(), (lab[:, 0] < 25).mean(), (lab[:, 0] > 85).mean(), sat.mean()],
    ])

    # --- shape (from the mask) ---
    H, W = fg.shape
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bw, bh = x1 - x0, y1 - y0
    box = fg[y0:y1, x0:x1]
    rows = box.mean(1)  # width profile per row
    cols = box.mean(0)  # height profile per column
    third = lambda p, k: float(p[k * len(p) // 3:(k + 1) * len(p) // 3].mean()) if len(p) >= 3 else float(p.mean())
    # Trousers: a column between the legs is mostly empty in the lower part. 1 = clear gap, 0 = none.
    bottom = box[int(bh * 0.6):] if bh > 6 else box
    center = bottom.mean(0)[int(bw * 0.3):max(int(bw * 0.7), int(bw * 0.3) + 1)]
    gap = 1.0 - float(center.min()) if center.size else 0.0
    shape = np.array([
        fg.mean(), bw / bh, box.mean(), xs.mean() / W, ys.mean() / H,
        (box == box[:, ::-1]).mean(), (box == box[::-1]).mean(),
        third(rows, 0), third(rows, 1), third(rows, 2), third(cols, 0), third(cols, 1), third(cols, 2), gap,
    ])

    # --- texture (pattern proxy): gradient magnitude inside the garment ---
    gray = rgb.astype(np.float64).mean(2)
    mag = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    inner = ndimage.binary_erosion(fg, iterations=2)
    m = mag[inner] if inner.any() else mag[fg]
    texture = np.array([(m > 40).mean(), m.mean() / 255])

    out = np.concatenate([color, shape, texture]).astype(np.float32)
    assert out.shape[0] == len(HANDCRAFTED_NAMES)
    return out
