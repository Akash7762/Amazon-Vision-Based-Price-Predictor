"""
Image preprocessing shared by the offline download/cache pipeline and, later,
the Phase 4 inference backend.

CRITICAL: the transform applied here at download time defines what the model
is trained on. The Phase 4 FastAPI backend MUST apply this same function to
uploaded images. If inference preprocessing diverges from training
preprocessing, the deployed app silently underperforms its offline metrics
with no error to signal it. That is why this lives in one importable module
rather than being inlined in the download script.

Resize strategy: letterbox-pad (chosen deliberately over stretch/crop).
Amazon catalog images sit on white backgrounds, so white padding blends into
the existing background, and preserving aspect ratio keeps the product's
proportions (long/thin vs square) intact — shape information that plausibly
correlates with product category and therefore price.
"""
from PIL import Image

# Matches the white background Amazon catalog shots already use.
LETTERBOX_FILL = (255, 255, 255)

# Identifier recorded in the cache manifest so a directory of images can be
# checked against the code that is about to write into it.
RESIZE_MODE = "letterbox_pad"


def letterbox_resize(img, size, fill=LETTERBOX_FILL, resample=Image.BILINEAR):
    """Resize `img` to a square `size` x `size` canvas, preserving aspect ratio.

    The image is scaled so its longer side equals `size`, then centred on a
    square `fill`-coloured canvas. Unlike a direct `.resize((size, size))`,
    this does not distort proportions.

    Args:
        img: a PIL Image (any mode; converted to RGB).
        size: target square edge length in pixels.
        fill: RGB tuple for the padding colour.
        resample: PIL resampling filter.

    Returns:
        A new RGB PIL Image of exactly (size, size).
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Image has a non-positive dimension: {img.size}")

    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), resample)

    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas
