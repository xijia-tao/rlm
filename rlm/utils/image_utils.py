"""
Image utilities for multimodal RLM support.

Handles image loading, base64 encoding, and building OpenAI-compatible
vision message content blocks.
"""

import base64
import io
from pathlib import Path
from typing import Any, Union


def load_image(source: Union[str, Path, Any]) -> Any:
    """Load and return a PIL Image from a file path or existing PIL Image."""
    from PIL import Image

    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    return source


def encode_image_base64(source: Union[str, Path, Any], fmt: str = "PNG") -> str:
    """Encode a PIL Image or image path to a base64 string."""
    from PIL import Image

    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    else:
        img = source

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_vision_messages(
    prompt: str,
    source: Union[str, Path, Any],
    fmt: str = "PNG",
) -> list[dict]:
    """
    Build an OpenAI-compatible message list for a vision query.

    Returns a list with a single user message whose content is a list of
    text + image_url blocks, ready to pass directly to the OpenAI client.
    """
    b64 = encode_image_base64(source, fmt=fmt)
    mime = f"image/{fmt.lower()}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]


def image_size_info(source: Union[str, Path, Any]) -> dict:
    """Return a dict with width, height, and mode for the image."""
    img = load_image(source)
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
    }
