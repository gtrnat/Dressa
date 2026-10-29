"""Background removal for single-garment photos (rembg, ONNX on CPU)."""

import os

import numpy as np
from PIL import Image

from app import config

# rembg reads its model directory from U2NET_HOME at import time.
os.environ.setdefault("U2NET_HOME", config.MODEL_CACHE_DIR)

from rembg import new_session, remove  # noqa: E402


class BackgroundRemover:
    def __init__(self, model_name: str = config.SEGMENTATION_MODEL):
        self.model_name = model_name
        # Downloads the ONNX weights on first run.
        self._session = new_session(model_name)

    def cutout(self, image: Image.Image) -> tuple[Image.Image, float]:
        """Return (RGBA cutout cropped to the garment, foreground ratio of the original frame).

        Blocking; call from a thread. Raises ValueError if nothing was segmented.
        """
        rgba: Image.Image = remove(image.convert("RGB"), session=self._session, post_process_mask=True)
        alpha = np.asarray(rgba.getchannel("A"))
        foreground = alpha > 127
        ratio = float(foreground.mean())
        if ratio < config.MIN_FOREGROUND_RATIO:
            raise ValueError("No garment detected in the photo")

        ys, xs = np.nonzero(foreground)
        pad = int(0.03 * max(rgba.size))
        box = (
            max(int(xs.min()) - pad, 0),
            max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad + 1, rgba.width),
            min(int(ys.max()) + pad + 1, rgba.height),
        )
        return rgba.crop(box), ratio
