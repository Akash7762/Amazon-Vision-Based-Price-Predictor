"""
Draw the app icons: a white lens with a dollar sign on teal.

    python frontend/scripts/make-icons.py      (needs Pillow; run from the repo root)

The design keeps everything inside the middle 80% circle, so the same image
works as a "maskable" icon (Android crops icons into circles and squircles)
and as a plain one. Writes:

    public/icons/icon-192.png, icon-512.png, icon-maskable-512.png   manifest
    app/icon.png                                                     browser tab
    app/apple-icon.png                                               iOS home screen
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRONTEND = Path(__file__).resolve().parent.parent
TEAL = (15, 118, 110)   # #0f766e, the app's theme colour
WHITE = (255, 255, 255)
FONTS = [Path("C:/Windows/Fonts/segoeuib.ttf"), Path("C:/Windows/Fonts/arialbd.ttf"),
         Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]


def draw(size):
    big = size * 4  # draw large, then shrink, for smooth edges
    img = Image.new("RGB", (big, big), TEAL)
    d = ImageDraw.Draw(img)
    r = big * 0.31
    c = big / 2
    d.ellipse((c - r, c - r, c + r, c + r), fill=WHITE)
    ring = big * 0.035
    d.ellipse((c - r + ring * 2, c - r + ring * 2, c + r - ring * 2, c + r - ring * 2),
              outline=TEAL, width=int(ring))
    font_path = next((f for f in FONTS if f.exists()), None)
    if font_path is None:
        sys.exit("No bold TrueType font found; add one to FONTS.")
    font = ImageFont.truetype(str(font_path), int(big * 0.36))
    d.text((c, c + big * 0.01), "$", font=font, fill=TEAL, anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def main():
    icons = FRONTEND / "public" / "icons"
    icons.mkdir(parents=True, exist_ok=True)
    outputs = {
        icons / "icon-192.png": 192,
        icons / "icon-512.png": 512,
        icons / "icon-maskable-512.png": 512,
        FRONTEND / "app" / "icon.png": 64,
        FRONTEND / "app" / "apple-icon.png": 180,
    }
    for path, size in outputs.items():
        draw(size).save(path, optimize=True)
        print(f"{path.relative_to(FRONTEND)}  {size}x{size}")


if __name__ == "__main__":
    main()
