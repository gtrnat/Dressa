"""Dressa ML Engine - FastAPI entrypoint.

Phase 1 stubs: /health, /process-image, /embed.
Model loading (YOLOv8-Seg / CLIP) will live in app/models, scoring in app/core.
"""

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

EMBEDDING_DIM = 512  # CLIP ViT-B/32
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Dressa ML Engine", version="0.1.0")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    models_loaded: bool


class ItemTags(BaseModel):
    category: str = Field(description="e.g. t-shirt, jacket, trousers")
    dominant_colors: list[str] = Field(description="Hex colors, most dominant first")
    pattern: str = Field(description="e.g. solid, striped, checked, floral")
    style: list[str] = Field(description="e.g. casual, formal, streetwear")


class ProcessImageResponse(BaseModel):
    filename: str
    mask_available: bool
    tags: ItemTags
    stub: bool = True


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vector: list[float]
    stub: bool = True


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return data


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ml-engine", models_loaded=False)


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(file: UploadFile = File(...)) -> ProcessImageResponse:
    """Background removal + auto-tagging. TODO: wire YOLOv8-Seg/SAM and a tagger."""
    await _read_image(file)
    return ProcessImageResponse(
        filename=file.filename or "upload",
        mask_available=False,
        tags=ItemTags(
            category="unknown",
            dominant_colors=[],
            pattern="unknown",
            style=[],
        ),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile = File(...)) -> EmbedResponse:
    """CLIP visual embedding. TODO: load CLIP ViT-B/32 (ONNX Runtime / PyTorch)."""
    await _read_image(file)
    return EmbedResponse(
        model="clip-vit-b-32",
        dim=EMBEDDING_DIM,
        vector=[0.0] * EMBEDDING_DIM,
    )
