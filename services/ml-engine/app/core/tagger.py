"""Zero-shot garment tagging with CLIP (image vs. text prompts in a shared space).

Label vocabularies are plain data so they can be tuned without retraining.
"""

from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from app import config

# item_type -> (ItemCategory, warmth 1..5, water_resistant). Keys are also the CLIP labels.
ITEM_TYPES: dict[str, tuple[str, int, bool]] = {
    "t-shirt": ("TOP", 1, False),
    "tank top": ("TOP", 1, False),
    "shirt": ("TOP", 2, False),
    "polo shirt": ("TOP", 2, False),
    "blouse": ("TOP", 2, False),
    "sweater": ("TOP", 4, False),
    "hoodie": ("TOP", 4, False),
    "jacket": ("OUTERWEAR", 4, False),
    "denim jacket": ("OUTERWEAR", 3, False),
    "blazer": ("OUTERWEAR", 3, False),
    "cardigan": ("OUTERWEAR", 3, False),
    "coat": ("OUTERWEAR", 5, False),
    "raincoat": ("OUTERWEAR", 3, True),
    "jeans": ("BOTTOM", 3, False),
    "trousers": ("BOTTOM", 3, False),
    "shorts": ("BOTTOM", 1, False),
    "skirt": ("BOTTOM", 2, False),
    "sweatpants": ("BOTTOM", 3, False),
    "dress": ("DRESS", 2, False),
    "jumpsuit": ("DRESS", 2, False),
    "sneakers": ("FOOTWEAR", 2, False),
    "formal leather shoes": ("FOOTWEAR", 2, False),
    "boots": ("FOOTWEAR", 4, True),
    "sandals": ("FOOTWEAR", 1, False),
    "hat": ("ACCESSORY", 2, False),
    "bag": ("ACCESSORY", 1, False),
    "scarf": ("ACCESSORY", 4, False),
    "belt": ("ACCESSORY", 1, False),
}

# Competing "not a single garment" labels, used to detect bad uploads.
EMPTY_LABEL = "an empty plain background"
NON_GARMENT_LABELS = (
    EMPTY_LABEL,
    "a group of people",
    "a person's face",
    "a building or street scene",
    "furniture or a household object",
)
NON_GARMENT_TEMPLATES = ("a photo of {}.",)

PATTERNS: dict[str, str] = {
    "solid": "a plain solid-colored {}",
    "striped": "a striped {}",
    "plaid": "a plaid checked {}",
    "floral": "a {} with a floral print",
    "polka dot": "a polka dot {}",
    "graphic print": "a {} with a graphic print or logo",
    "camouflage": "a camouflage {}",
    "animal print": "a {} with an animal print",
}

STYLES: dict[str, str] = {
    "casual": "a casual everyday {}",
    "formal": "a formal elegant {}",
    "business": "a business office {}",
    "streetwear": "a streetwear {}",
    "sporty": "a sporty athletic {}",
    "vintage": "a vintage retro {}",
    "minimalist": "a minimalist basic {}",
    "bohemian": "a bohemian {}",
}

ITEM_TEMPLATES = ("a photo of a {}.", "a product photo of a {}, clothing item on a white background.")
CLIP_LOGIT_SCALE = 100.0  # CLIP's learned temperature
STYLE_MIN_PROB = 0.2
STYLE_MAX_LABELS = 2


@dataclass
class Prediction:
    label: str
    confidence: float


@dataclass
class Tags:
    garment_probability: float  # P(any garment label) vs. non-garment labels
    is_empty: bool  # best overall label is "empty background"
    item_type: Prediction
    category: str
    warmth_level: int
    water_resistant: bool
    pattern: Prediction
    styles: list[Prediction]


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class ZeroShotTagger:
    def __init__(self, model_name: str = config.CLIP_TEXT_MODEL, cache_dir: str = config.MODEL_CACHE_DIR):
        self._text = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._item_labels = list(ITEM_TYPES)
        self._item_matrix = self._encode_ensemble(self._item_labels, ITEM_TEMPLATES)
        self._non_garment_matrix = self._encode_ensemble(list(NON_GARMENT_LABELS), NON_GARMENT_TEMPLATES)
        self._pattern_cache: dict[str, np.ndarray] = {}
        self._style_cache: dict[str, np.ndarray] = {}

    def _encode(self, prompts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(list(self._text.embed(prompts)), dtype=np.float32))

    def _encode_ensemble(self, labels: list[str], templates: tuple[str, ...]) -> np.ndarray:
        """Average the prompt embeddings of each label over several templates."""
        per_template = [self._encode([t.format(label) for label in labels]) for t in templates]
        return _normalize(np.mean(per_template, axis=0))

    def _attribute_matrix(self, cache: dict[str, np.ndarray], vocab: dict[str, str], item_type: str) -> np.ndarray:
        # Prompts mention the detected garment ("a striped shirt") which helps CLIP a lot.
        if item_type not in cache:
            cache[item_type] = self._encode([tpl.format(item_type) for tpl in vocab.values()])
        return cache[item_type]

    def _probs(self, image_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return _softmax(CLIP_LOGIT_SCALE * (matrix @ image_vec))

    def tag(self, image_vector: list[float], item_type_hint: str | None = None) -> Tags:
        """image_vector: L2-normalized CLIP image embedding. Blocking; call from a thread.

        item_type_hint: garment type from the trained ML classifier; used in the pattern/style
        prompts ("a striped t-shirt") instead of CLIP's own item-type guess.
        """
        v = np.asarray(image_vector, dtype=np.float32)

        n_items = len(self._item_labels)
        all_p = self._probs(v, np.vstack([self._item_matrix, self._non_garment_matrix]))
        garment_probability = float(all_p[:n_items].sum())
        is_empty = int(all_p.argmax()) == n_items + NON_GARMENT_LABELS.index(EMPTY_LABEL)

        item_p = self._probs(v, self._item_matrix)  # confidence among garment types only
        i = int(item_p.argmax())
        item_type = self._item_labels[i]
        category, warmth, water_resistant = ITEM_TYPES[item_type]
        prompt_item = item_type_hint or item_type

        pattern_labels = list(PATTERNS)
        pattern_p = self._probs(v, self._attribute_matrix(self._pattern_cache, PATTERNS, prompt_item))
        j = int(pattern_p.argmax())

        style_labels = list(STYLES)
        style_p = self._probs(v, self._attribute_matrix(self._style_cache, STYLES, prompt_item))
        order = np.argsort(-style_p)
        styles = [Prediction(style_labels[k], round(float(style_p[k]), 3)) for k in order[:STYLE_MAX_LABELS]
                  if style_p[k] >= STYLE_MIN_PROB]
        if not styles:  # always return at least the best guess
            styles = [Prediction(style_labels[order[0]], round(float(style_p[order[0]]), 3))]

        return Tags(
            garment_probability=round(garment_probability, 3),
            is_empty=is_empty,
            item_type=Prediction(item_type, round(float(item_p[i]), 3)),
            category=category,
            warmth_level=warmth,
            water_resistant=water_resistant,
            pattern=Prediction(pattern_labels[j], round(float(pattern_p[j]), 3)),
            styles=styles,
        )
