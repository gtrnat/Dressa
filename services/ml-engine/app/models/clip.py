"""Image embedder (ONNX via FastEmbed, CPU). Default: CLIP ViT-B/32."""

import numpy as np
from fastembed import ImageEmbedding
from PIL import Image

from app import config


def pad_to_square(image: Image.Image) -> Image.Image:
    """CLIP center-crops non-square inputs; pad with white instead so no part of the garment is cut off."""
    image = image.convert("RGB")
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


class ClipImageEmbedder:
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        cache_dir: str = config.MODEL_CACHE_DIR,
        expected_dim: int | None = config.EMBEDDING_DIM,
    ):
        self.model_name = model_name
        self.expected_dim = expected_dim
        # Downloads the ONNX weights on first run, then loads from cache_dir.
        self._model = ImageEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalized embedding. Blocking; call from a thread."""
        vector = np.asarray(next(iter(self._model.embed([pad_to_square(image)]))), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        if self.expected_dim is not None and vector.shape[0] != self.expected_dim:
            raise RuntimeError(f"Expected {self.expected_dim}-dim vector, got {vector.shape[0]}")
        return vector.tolist()
