"""CLIP ViT-B/32 image embedder (ONNX via FastEmbed, CPU)."""

import numpy as np
from fastembed import ImageEmbedding
from PIL import Image

from app import config


class ClipImageEmbedder:
    def __init__(self, model_name: str = config.CLIP_MODEL, cache_dir: str = config.MODEL_CACHE_DIR):
        self.model_name = model_name
        # Downloads the ONNX weights on first run, then loads from cache_dir.
        self._model = ImageEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, image: Image.Image) -> list[float]:
        """Return an L2-normalized 512-dim embedding. Blocking; call from a thread."""
        vector = next(iter(self._model.embed([image])))
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        if vector.shape[0] != config.EMBEDDING_DIM:
            raise RuntimeError(f"Expected {config.EMBEDDING_DIM}-dim vector, got {vector.shape[0]}")
        return vector.tolist()
