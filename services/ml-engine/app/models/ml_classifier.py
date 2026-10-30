"""Dressa ML attribute classifier (scikit-learn, trained in ml/training/dressa_ml_classifier.ipynb).

Pretrained deep embedding (frozen feature extractor) + handcrafted color/shape features
-> one numeric feature vector -> one scikit-learn model per head (item_type, style, season).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from PIL import Image

from app import config
from app.core.features import FEATURE_VERSION, HANDCRAFTED_NAMES, handcrafted, l2_normalize, prepare
from app.models.clip import ClipImageEmbedder

log = logging.getLogger(__name__)

BUNDLE_FORMAT = "dressa-ml-v1"
REQUIRED_HEADS = ("item_type", "style", "season")
VALID_CATEGORIES = {"TOP", "BOTTOM", "OUTERWEAR", "DRESS", "FOOTWEAR", "ACCESSORY"}

# item_type -> (warmth 1..5, water_resistant). Category comes from the bundle.
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


class MLAttributeClassifier:
    def __init__(self, bundle: dict, embedder: ClipImageEmbedder | None):
        self._validate(bundle)
        self.bundle = bundle
        self.feature_set: str = bundle["feature_set"]
        self.heads: dict = bundle["heads"]
        self.item_to_category: dict[str, str] = bundle["item_type_to_category"]
        self.model_names = {h: self.heads[h]["name"] for h in REQUIRED_HEADS}
        self._embedder = embedder
        if self.feature_set != "handcrafted" and embedder is None:
            raise ValueError("Bundle needs an embedding extractor but none was provided")

        # Fail at startup (not on the first request) if the bundle and the feature code disagree.
        probe = self.features(Image.new("RGB", (60, 80), (255, 255, 255)))
        for head in REQUIRED_HEADS:
            n_in = getattr(self.heads[head]["model"], "n_features_in_", probe.shape[0])
            if n_in != probe.shape[0]:
                raise ValueError(f"Head {head} expects {n_in} features, feature code produces {probe.shape[0]}")
        unknown = set(self.heads["item_type"]["classes"]) - set(ITEM_WARMTH)
        if unknown:
            log.warning("No warmth mapping for item types %s; using default %s", sorted(unknown), DEFAULT_WARMTH)

    @staticmethod
    def _validate(b: dict) -> None:
        if b.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"Not a {BUNDLE_FORMAT} bundle")
        if b.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"Bundle feature_version {b.get('feature_version')} != code FEATURE_VERSION {FEATURE_VERSION}")
        if b.get("handcrafted_names") and list(b["handcrafted_names"]) != HANDCRAFTED_NAMES:
            raise ValueError("Handcrafted feature names differ between bundle and code")
        if b.get("sklearn_version") != sklearn.__version__:
            raise ValueError(f"Bundle was trained with scikit-learn {b.get('sklearn_version')}, "
                             f"ml-engine has {sklearn.__version__}; install the same version")
        for head in REQUIRED_HEADS:
            if head not in b.get("heads", {}):
                raise ValueError(f"Bundle has no '{head}' head")
        cats = b.get("item_type_to_category", {})
        if set(b["heads"]["item_type"]["classes"]) - set(cats) or set(cats.values()) - VALID_CATEGORIES:
            raise ValueError("item_type_to_category must cover every item_type with a valid category")

    def features(self, image: Image.Image) -> np.ndarray:
        prepared = prepare(image)
        parts = []
        if self.feature_set in ("embedding", "embedding+handcrafted"):
            parts.append(l2_normalize(np.asarray(self._embedder.embed(prepared))))
        if self.feature_set in ("handcrafted", "embedding+handcrafted"):
            parts.append(handcrafted(prepared))
        return np.concatenate(parts).astype(np.float32)

    def predict(self, image: Image.Image) -> AttributePrediction:
        """Blocking; call from a thread."""
        x = self.features(image)[None]
        heads: dict[str, HeadPrediction] = {}
        for head in REQUIRED_HEADS:
            model, classes = self.heads[head]["model"], self.heads[head]["classes"]
            proba = model.predict_proba(x)[0]
            i = int(np.argmax(proba))
            heads[head] = HeadPrediction(classes[int(model.classes_[i])], round(float(proba[i]), 3))

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


def load_ml_classifier(clip_embedder: ClipImageEmbedder) -> MLAttributeClassifier | None:
    """Returns None (CLIP zero-shot fallback) if the trained bundle is not present.

    Security: joblib files can execute code when loaded. Only load bundles you trained yourself.
    """
    path = Path(config.ML_MODEL_PATH)
    if not path.exists():
        log.warning("ML classifier not found at %s; falling back to CLIP zero-shot tagging", path)
        return None
    bundle = joblib.load(path)
    extractor = bundle.get("extractor")
    if bundle.get("feature_set") == "handcrafted":
        embedder = None
    elif extractor == clip_embedder.model_name:
        embedder = clip_embedder  # reuse the CLIP model already loaded for /embed
    else:
        embedder = ClipImageEmbedder(model_name=extractor, expected_dim=None)
    clf = MLAttributeClassifier(bundle, embedder)
    log.info("Loaded ML classifier from %s (extractor=%s, features=%s, models=%s)",
             path, extractor, clf.feature_set, clf.model_names)
    return clf


def describe(clf: MLAttributeClassifier) -> dict:
    meta = {k: clf.bundle.get(k) for k in ("extractor", "feature_set", "trained_at", "metrics")}
    return json.loads(json.dumps({**meta, "models": clf.model_names}, default=str))
