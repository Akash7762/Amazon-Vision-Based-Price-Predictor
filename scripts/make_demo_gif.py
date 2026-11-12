"""
Makes docs/images/demo.gif, the README's demo, from the frames that
e2e/media/screenshots.spec.ts saves in e2e/test-results/demo-frames/.
Pillow only.

    cd e2e && npx playwright test -c media.config.ts     # screenshots and frames
    python scripts/make_demo_gif.py                       # repo root
"""
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FRAMES = REPO / "e2e" / "test-results" / "demo-frames"
OUT = REPO / "docs" / "images" / "demo.gif"
HOLD_MS = [1800, 1200, 4000]  # start screen, "Estimating the price…", the result


def main():
    paths = sorted(FRAMES.glob("*.png"))
    if len(paths) != len(HOLD_MS):
        raise SystemExit(f"expected {len(HOLD_MS)} frames in {FRAMES}, found {len(paths)}: "
                         "run the screenshots first (see the docstring)")
    frames = [Image.open(p).convert("RGB") for p in paths]

    # One palette for every frame, so the colours don't shift between them;
    # no dithering, which would speckle the flat colours of the page.
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for i, frame in enumerate(frames):
        strip.paste(frame, (0, i * frame.height))
    palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    gif[0].save(OUT, save_all=True, append_images=gif[1:], duration=HOLD_MS, loop=0, optimize=True)
    print(f"{OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(gif)} frames)")


if __name__ == "__main__":
    main()
