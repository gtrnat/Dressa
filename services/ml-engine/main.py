"""Dressa ML Engine - FastAPI entrypoint.

Internal service: called by api-gateway, not exposed to clients directly
(no auth here; user_id is trusted from the gateway).
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import config
from app.core.colors import dominant_colors
from app.core.tagger import ZeroShotTagger
from app.core.vector_store import VectorStore
from app.images import decode_image, downscale, flatten_on_white, open_image, png_base64, read_upload
from app.models.attribute_classifier import load_attribute_classifier
from app.models.clip import ClipImageEmbedder
from app.models.segmenter import BackgroundRemover

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ml-engine")

NEEDS_REVIEW_BELOW = 0.5  # item-type confidence under this -> ask the user to confirm


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models (first run downloads the weights)...")
    app.state.embedder = await run_in_threadpool(ClipImageEmbedder)
    app.state.tagger = await run_in_threadpool(ZeroShotTagger)
    app.state.segmenter = await run_in_threadpool(BackgroundRemover)
    # DressaNet replaces CLIP for item type / style / warmth when its files are present.
    app.state.attributes = await run_in_threadpool(load_attribute_classifier)
    log.info("Models loaded: %s, %s, %s; attribute model: %s", config.CLIP_MODEL, config.CLIP_TEXT_MODEL,
             config.SEGMENTATION_MODEL, _attribute_model_name())
    app.state.vectors = VectorStore()
    try:
        await app.state.vectors.ensure_collection()
    except Exception as exc:  # Qdrant down at startup: keep serving, retry lazily
        log.warning("Qdrant not reachable at %s: %s", config.QDRANT_URL, exc)
    yield
    await app.state.vectors.close()


app = FastAPI(title="Dressa ML Engine", version="0.3.0", lifespan=lifespan)

AttributeModel = Literal["dressanet", "clip-zero-shot"]


def _attribute_model_name() -> AttributeModel:
    return "dressanet" if getattr(app.state, "attributes", None) is not None else "clip-zero-shot"


# ---------- Schemas ----------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    models_loaded: bool
    attribute_model: AttributeModel
    qdrant_reachable: bool


ItemCategory = Literal["TOP", "BOTTOM", "OUTERWEAR", "DRESS", "FOOTWEAR", "ACCESSORY"]


class ColorTag(BaseModel):
    hex: str
    name: str
    ratio: float = Field(description="Share of garment pixels, 0..1")


class LabelTag(BaseModel):
    label: str
    confidence: float = Field(description="Model probability, 0..1")


class ItemTags(BaseModel):
    category: ItemCategory = Field(description="Maps to WardrobeItem.category")
    item_type: LabelTag = Field(description="e.g. t-shirt, jacket, jeans")
    dominant_colors: list[ColorTag] = Field(description="Most dominant first")
    pattern: LabelTag = Field(description="e.g. solid, striped, plaid, floral")
    styles: list[LabelTag] = Field(description="e.g. casual, formal, sporty")
    season: LabelTag | None = Field(default=None, description="summer/fall/winter/spring (DressaNet only)")
    warmth_level: int = Field(ge=1, le=5, description="Derived from item_type (and season for DressaNet)")
    water_resistant: bool


class ProcessImageResponse(BaseModel):
    filename: str
    background_removed: bool
    foreground_ratio: float | None = Field(description="Garment area / original frame area")
    cutout_png_base64: str = Field(description="Transparent PNG cropped to the garment")
    cutout_width: int
    cutout_height: int
    tags: ItemTags
    attribute_model: AttributeModel = Field(description="Model that produced item_type/style/warmth")
    garment_probability: float = Field(
        description="How likely the cutout is a single garment (vs. people, scenes, objects), 0..1"
    )
    needs_review: bool = Field(
        description="Low-confidence result (unclear garment type, or the photo is not a single garment)"
    )


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vector: list[float]
    stored: bool = Field(description="True if the vector was upserted to Qdrant")
    qdrant_vector_id: UUID | None = None


class SimilarItem(BaseModel):
    item_id: UUID
    score: float = Field(description="Cosine similarity, -1..1 (higher = more similar)")
    category: str | None = None


class SimilarResponse(BaseModel):
    item_id: UUID
    results: list[SimilarItem]


def _vectors() -> VectorStore:
    return app.state.vectors


async def _qdrant_call(coro):
    try:
        return await coro
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Qdrant error")
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc


# ---------- Routes ----------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="ml-engine",
        models_loaded=all(
            getattr(app.state, name, None) is not None for name in ("embedder", "tagger", "segmenter")
        ),
        attribute_model=_attribute_model_name(),
        qdrant_reachable=await _vectors().is_reachable(),
    )


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(
    file: Annotated[UploadFile, File(description="Photo of a single garment")],
    remove_background: Annotated[bool, Form(description="Set false if the upload is already a cutout")] = True,
) -> ProcessImageResponse:
    """Background removal + auto-tagging (category, type, colors, pattern, style, warmth)."""
    image = open_image(await read_upload(file))

    ratio: float | None = None
    if remove_background:
        try:
            cutout, ratio = await run_in_threadpool(
                app.state.segmenter.cutout, downscale(image, config.SEGMENT_MAX_SIDE)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        cutout = image.convert("RGBA")
    cutout = downscale(cutout, config.CUTOUT_MAX_SIDE)

    colors = await run_in_threadpool(dominant_colors, cutout)
    vector = await run_in_threadpool(app.state.embedder.embed, flatten_on_white(cutout))

    # DressaNet (own model): item type, category, style, season, warmth.
    # CLIP is still used for the "is this a garment?" check and for pattern (no pattern labels in training data).
    attrs = None
    if app.state.attributes is not None:
        attrs = await run_in_threadpool(app.state.attributes.predict, cutout)
    clip = await run_in_threadpool(app.state.tagger.tag, vector, attrs.item_type.label if attrs else None)
    if clip.is_empty:
        raise HTTPException(status_code=422, detail="No garment detected in the photo")

    if attrs is not None:
        item_type = LabelTag(label=attrs.item_type.label, confidence=attrs.item_type.confidence)
        tags = ItemTags(
            category=attrs.category,
            item_type=item_type,
            dominant_colors=[ColorTag(hex=c.hex, name=c.name, ratio=c.ratio) for c in colors],
            pattern=LabelTag(label=clip.pattern.label, confidence=clip.pattern.confidence),
            styles=[LabelTag(label=attrs.style.label, confidence=attrs.style.confidence)],
            season=LabelTag(label=attrs.season.label, confidence=attrs.season.confidence),
            warmth_level=attrs.warmth_level,
            water_resistant=attrs.water_resistant,
        )
    else:
        item_type = LabelTag(label=clip.item_type.label, confidence=clip.item_type.confidence)
        tags = ItemTags(
            category=clip.category,
            item_type=item_type,
            dominant_colors=[ColorTag(hex=c.hex, name=c.name, ratio=c.ratio) for c in colors],
            pattern=LabelTag(label=clip.pattern.label, confidence=clip.pattern.confidence),
            styles=[LabelTag(label=s.label, confidence=s.confidence) for s in clip.styles],
            warmth_level=clip.warmth_level,
            water_resistant=clip.water_resistant,
        )

    return ProcessImageResponse(
        filename=file.filename or "upload",
        background_removed=remove_background,
        foreground_ratio=round(ratio, 4) if ratio is not None else None,
        cutout_png_base64=await run_in_threadpool(png_base64, cutout),
        cutout_width=cutout.width,
        cutout_height=cutout.height,
        tags=tags,
        attribute_model=_attribute_model_name(),
        garment_probability=clip.garment_probability,
        needs_review=item_type.confidence < NEEDS_REVIEW_BELOW or clip.garment_probability < NEEDS_REVIEW_BELOW,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    file: Annotated[UploadFile, File()],
    item_id: Annotated[UUID | None, Form(description="WardrobeItem.id; set with user_id to store in Qdrant")] = None,
    user_id: Annotated[UUID | None, Form()] = None,
    category: Annotated[str | None, Form(max_length=32)] = None,
) -> EmbedResponse:
    """CLIP ViT-B/32 image embedding (512-dim, L2-normalized).

    Without item_id/user_id: returns the vector only.
    With both: also upserts it to Qdrant under point id = item_id.
    """
    if (item_id is None) != (user_id is None):
        raise HTTPException(status_code=422, detail="item_id and user_id must be provided together")

    image = decode_image(await read_upload(file))
    vector = await run_in_threadpool(app.state.embedder.embed, image)

    stored = False
    if item_id and user_id:
        await _qdrant_call(_vectors().upsert(item_id, user_id, vector, category))
        stored = True

    return EmbedResponse(
        model=config.CLIP_MODEL,
        dim=len(vector),
        vector=vector,
        stored=stored,
        qdrant_vector_id=item_id if stored else None,
    )


@app.get("/items/{item_id}/similar", response_model=SimilarResponse)
async def similar_items(
    item_id: UUID,
    user_id: Annotated[UUID, Query()],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    category: Annotated[str | None, Query(max_length=32)] = None,
) -> SimilarResponse:
    """Most visually similar items in the same user's wardrobe."""
    if not await _qdrant_call(_vectors().exists_for_user(item_id, user_id)):
        raise HTTPException(status_code=404, detail="Item embedding not found")
    hits = await _qdrant_call(_vectors().similar(item_id, user_id, limit, category))
    return SimilarResponse(
        item_id=item_id,
        results=[SimilarItem(item_id=pid, score=score, category=p.get("category")) for pid, score, p in hits],
    )


@app.delete("/items/{item_id}", status_code=204)
async def delete_item_embedding(item_id: UUID, user_id: Annotated[UUID, Query()]) -> None:
    if not await _qdrant_call(_vectors().exists_for_user(item_id, user_id)):
        raise HTTPException(status_code=404, detail="Item embedding not found")
    await _qdrant_call(_vectors().delete(item_id))
