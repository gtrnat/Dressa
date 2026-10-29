"""Dominant color extraction from a garment cutout (k-means in CIELAB)."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Fashion-oriented color names (sRGB). Nearest match is done in CIELAB.
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "charcoal": (54, 69, 79),
    "gray": (128, 128, 128),
    "light gray": (200, 200, 200),
    "white": (245, 245, 245),
    "cream": (240, 230, 200),
    "beige": (215, 195, 160),
    "khaki": (189, 170, 120),
    "brown": (110, 70, 40),
    "tan": (190, 140, 90),
    "red": (200, 30, 40),
    "maroon": (110, 20, 35),
    "pink": (240, 160, 180),
    "orange": (240, 130, 40),
    "yellow": (245, 210, 50),
    "mustard": (200, 160, 40),
    "olive": (110, 115, 50),
    "green": (40, 140, 70),
    "dark green": (25, 70, 45),
    "teal": (20, 130, 130),
    "light blue": (150, 190, 230),
    "blue": (40, 90, 200),
    "denim": (70, 100, 140),
    "navy": (25, 35, 75),
    "purple": (110, 50, 140),
    "lavender": (190, 170, 220),
}


@dataclass
class DominantColor:
    hex: str
    name: str
    ratio: float


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) uint8/float in 0..255 -> (N, 3) CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


_NAMES = list(NAMED_COLORS)
_NAMED_LAB = _srgb_to_lab(np.array(list(NAMED_COLORS.values())))


def _kmeans(points: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() == 0:
            break
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(len(centers))])
        if np.allclose(new, centers, atol=0.5):
            break
        centers = new
    return centers, labels


def dominant_colors(cutout: Image.Image, k: int = 5, min_ratio: float = 0.05) -> list[DominantColor]:
    """Most dominant first. Only opaque (foreground) pixels are considered."""
    img = cutout.convert("RGBA")
    img.thumbnail((128, 128))
    px = np.asarray(img).reshape(-1, 4)
    rgb = px[px[:, 3] > 127, :3]
    if len(rgb) == 0:
        return []

    lab = _srgb_to_lab(rgb)
    centers, labels = _kmeans(lab, k=min(k, len(np.unique(rgb, axis=0))))
    counts = np.bincount(labels, minlength=len(centers))

    results: dict[str, DominantColor] = {}
    for i in np.argsort(-counts):
        ratio = counts[i] / counts.sum()
        if ratio < min_ratio:
            continue
        mean_rgb = rgb[labels == i].mean(0).round().astype(int)
        name = _NAMES[int(np.argmin(((_NAMED_LAB - centers[i]) ** 2).sum(1)))]
        if name in results:  # merge clusters that map to the same name
            results[name].ratio += float(ratio)
            continue
        results[name] = DominantColor(hex="#{:02x}{:02x}{:02x}".format(*mean_rgb), name=name, ratio=float(ratio))

    ordered = sorted(results.values(), key=lambda c: -c.ratio)
    for c in ordered:
        c.ratio = round(c.ratio, 3)
    return ordered
