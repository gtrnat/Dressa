"""DressaNet attribute classifier (trained from scratch, see ml/training notebook).

Loads the ONNX model + labels JSON exported by the notebook and predicts
item_type / style / season from a garment cutout.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app import config
from app.images import flatten_on_white

log = logging.getLogger(__name__)

REQUIRED_HEADS = ("item_type", "style", "season")
VALID_CATEGORIES = {"TOP", "BOTTOM", "OUTERWEAR", "DRESS", "FOOTWEAR", "ACCESSORY"}

# DressaNet item_type -> (warmth 1..5, water_resistant). Category comes from the labels JSON.
ITEM_WARMTH: dict[str, tuple[int, bool]] = {
    "t-shirt": (1, False), "shirt": (2, False), "top": (1, False), "kurta": (2, False),
    "sweatshirt": (4, False), "sweater": (4, False), "jacket": (4, False),
    "jeans": (3, False), "trousers": (3, False), "shorts": (1, False), "track pants": (3, False),
    "leggings": (2, False), "skirt": (2, False), "dress": (2, False),
    "sneakers": (2, False), "formal shoes": (2, False), "heels": (1, False), "flats": (1, False),
    "sandals": (1, False), "bag": (1, False), "cap": (2, False), "belt": (1, False), "scarf": (4, False),
}
DEFAULT_WARMTH = (3, False)
SEASON_CONFIDENT = 0.6  # season prob above this nudges warmth by +-1


@dataclass
class HeadPrediction:
    label: str
    confidence: float


@dataclass
class AttributePrediction:
    item_type: HeadPrediction
    category: str
    style: HeadPrediction
    season: HeadPrediction
    warmth_level: int
    water_resistant: bool


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class AttributeClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        self._validate(meta)
        self.meta = meta
        self.img_size = int(meta["img_size"])
        self.classes: dict[str, list[str]] = meta["classes"]
        self.item_to_category: dict[str, str] = meta["item_type_to_category"]
        self._mean = np.asarray(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(meta["std"], dtype=np.float32).reshape(3, 1, 1)

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]
        missing = set(REQUIRED_HEADS) - set(self._outputs)
        if missing:
            raise ValueError(f"ONNX model is missing outputs {sorted(missing)} (has {self._outputs})")

        # Fail at startup, not on the first request, if shapes don't match the labels file.
        probe = self._run(np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32))
        for head in REQUIRED_HEADS:
            if probe[head].shape[-1] != len(self.classes[head]):
                raise ValueError(f"Head {head}: model has {probe[head].shape[-1]} outputs, "
                                 f"labels file has {len(self.classes[head])} classes")
        unknown = set(self.classes["item_type"]) - set(ITEM_WARMTH)
        if unknown:
            log.warning("No warmth mapping for item types %s; using default %s", sorted(unknown), DEFAULT_WARMTH)

    @staticmethod
    def _validate(meta: dict) -> None:
        for key in ("img_size", "mean", "std", "classes", "item_type_to_category"):
            if key not in meta:
                raise ValueError(f"Labels file is missing '{key}'")
        for head in REQUIRED_HEADS:
            if not meta["classes"].get(head):
                raise ValueError(f"Labels file has no classes for head '{head}'")
        bad = {c for c in meta["item_type_to_category"].values() if c not in VALID_CATEGORIES}
        if bad or set(meta["classes"]["item_type"]) - set(meta["item_type_to_category"]):
            raise ValueError("item_type_to_category must cover every item_type with a valid category")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Same steps as the notebook's preprocess(): white background, pad to square, resize, normalize."""
        img = flatten_on_white(image)
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        canvas = canvas.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self._mean) / self._std)[None].astype(np.float32)

    def _run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(self._outputs, self._session.run(None, {self._input: x})))

    def predict(self, image: Image.Image) -> AttributePrediction:
        """Blocking; call from a thread."""
        out = self._run(self.preprocess(image))
        heads: dict[str, HeadPrediction] = {}
        for head in REQUIRED_HEADS:
            p = _softmax(out[head][0].astype(np.float64))
            i = int(p.argmax())
            heads[head] = HeadPrediction(self.classes[head][i], round(float(p[i]), 3))

        item_type = heads["item_type"].label
        warmth, water_resistant = ITEM_WARMTH.get(item_type, DEFAULT_WARMTH)
        season = heads["season"]
        if season.confidence >= SEASON_CONFIDENT:
            if season.label == "winter":
                warmth = min(warmth + 1, 5)
            elif season.label == "summer":
                warmth = max(warmth - 1, 1)

        return AttributePrediction(
            item_type=heads["item_type"],
            category=self.item_to_category[item_type],
            style=heads["style"],
            season=season,
            warmth_level=warmth,
            water_resistant=water_resistant,
        )


def load_attribute_classifier() -> AttributeClassifier | None:
    """Returns None (CLIP zero-shot fallback) if the exported files are not present."""
    model_path, labels_path = Path(config.ATTRIBUTE_MODEL_PATH), Path(config.ATTRIBUTE_LABELS_PATH)
    if not model_path.exists() or not labels_path.exists():
        log.warning("DressaNet not found (%s, %s); falling back to CLIP zero-shot tagging", model_path, labels_path)
        return None
    clf = AttributeClassifier(model_path, labels_path)
    log.info("Loaded DressaNet from %s (%d item types)", model_path, len(clf.classes["item_type"]))
    return clf
