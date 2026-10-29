"""Upload validation and image decoding."""

import base64
import io

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


async def read_upload(file: UploadFile) -> bytes:
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type. Allowed: {sorted(config.ALLOWED_CONTENT_TYPES)}",
        )
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return data


def open_image(data: bytes) -> Image.Image:
    """Decode bytes, apply EXIF orientation (phone photos), keep alpha if present."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    return img.convert("RGBA" if has_alpha else "RGB")


def flatten_on_white(img: Image.Image) -> Image.Image:
    """RGBA -> RGB with transparent areas on white, so cutouts embed consistently."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.getchannel("A"))
    return background


def decode_image(data: bytes) -> Image.Image:
    return flatten_on_white(open_image(data))


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
