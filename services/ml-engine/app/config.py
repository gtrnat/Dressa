import os
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "wardrobe_items")

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(SERVICE_ROOT / ".cache" / "models"))
CLIP_MODEL = os.getenv("CLIP_MODEL", "Qdrant/clip-ViT-B-32-vision")
CLIP_TEXT_MODEL = os.getenv("CLIP_TEXT_MODEL", "Qdrant/clip-ViT-B-32-text")  # same space as CLIP_MODEL
EMBEDDING_DIM = 512
# rembg model: isnet-general-use (default), u2net, u2netp (small/fast), birefnet-general (best, ~1 GB)
SEGMENTATION_MODEL = os.getenv("SEGMENTATION_MODEL", "isnet-general-use")

# DressaNet (trained in ml/training). If missing, tagging falls back to CLIP zero-shot.
WEIGHTS_DIR = SERVICE_ROOT / "weights"
ATTRIBUTE_MODEL_PATH = os.getenv("ATTRIBUTE_MODEL_PATH", str(WEIGHTS_DIR / "dressa_attr.onnx"))
ATTRIBUTE_LABELS_PATH = os.getenv("ATTRIBUTE_LABELS_PATH", str(WEIGHTS_DIR / "dressa_attr_labels.json"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP, guards against decompression bombs
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

SEGMENT_MAX_SIDE = 1024  # input is downscaled before segmentation
CUTOUT_MAX_SIDE = 1024  # returned PNG cutout
MIN_FOREGROUND_RATIO = 0.01  # below this the photo is treated as "no garment found"
