"""
Multimodal RLM example: answering questions about a high-resolution image.

The RLM receives an image as context (via ImageContext). The image is loaded
as a PIL Image into the REPL as `context`, but is NOT sent to the VLM upfront.
The VLM inspects `context.size`, resizes/crops regions with PIL, then calls
`view_image(region, prompt)` to send a (cheaper, smaller) image to itself and
get a text response. This avoids burning a large number of image tokens on an
image that may not need to be seen at full resolution.

Requirements:
    pip install pillow
    export OPENAI_API_KEY=...  (or set in .env)

The model used must support vision (e.g. gpt-4o, gpt-4o-mini).
"""

import os

from dotenv import load_dotenv
from PIL import Image, ImageDraw

from rlm import RLM, ImageContext

load_dotenv()

# ── Demo: create a synthetic test image if no real one is provided ──────────

def make_demo_image(path: str = "/tmp/demo_scene.png") -> str:
    """Create a simple synthetic image with shapes and text for testing."""
    img = Image.new("RGB", (800, 600), color=(240, 240, 255))
    draw = ImageDraw.Draw(img)

    # Red rectangle top-left
    draw.rectangle([50, 50, 250, 200], fill=(220, 60, 60), outline=(180, 20, 20), width=3)
    draw.text((100, 110), "RED ZONE", fill="white")

    # Blue circle center
    draw.ellipse([300, 200, 500, 400], fill=(60, 120, 220), outline=(20, 60, 180), width=3)
    draw.text((350, 285), "BLUE\nCIRCLE", fill="white")

    # Green triangle bottom-right
    draw.polygon([(600, 500), (750, 300), (800, 500)], fill=(60, 200, 80), outline=(20, 140, 40))
    draw.text((680, 430), "GREEN", fill="white")

    # Labels
    draw.text((10, 10), "Test scene: geometric shapes", fill=(50, 50, 50))
    draw.text((10, 560), "Bottom bar: 800x600 synthetic image", fill=(80, 80, 80))

    img.save(path)
    print(f"Demo image saved: {path}")
    return path


# ── Main ────────────────────────────────────────────────────────────────────

image_path = os.environ.get("IMAGE_PATH") or make_demo_image()

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-4o-mini",  # any vision-capable model
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    environment="local",
    max_depth=1,
    max_iterations=10,
    verbose=True,
)

result = rlm.completion(
    ImageContext(image_path),
    root_prompt=(
        "Describe all the shapes and colors you see. "
        "Then count how many distinct colored regions there are."
    ),
)
# The VLM will not see the image until it calls view_image() from the REPL, e.g.:
#   w, h = context.size
#   overview = view_image(context.resize((512, 512)), "Describe all shapes and colors.")
#   region = context.crop((0, 0, w//2, h//2))
#   detail = view_image(region, "What colored shapes are in this region?")

print("\n=== Final Answer ===")
print(result.response)
