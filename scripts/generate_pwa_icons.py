# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Generate the PWA + favicon icon set from the master art.

Source: design/icon-master.png (the teal rounded-square "á" glyph). The master has
opaque near-white corners around the rounded square; we flood-fill those to
transparent so the square floats, then derive every shipped size:

  uv run scripts/generate_pwa_icons.py

Outputs (committed, small) under public/:
  * icon-192.png / icon-512.png            manifest `purpose: any` (transparent corners)
  * icon-192-maskable.png / -512-maskable  manifest `purpose: maskable` (teal to edges,
                                           glyph inside the 80% Android safe zone)
  * apple-touch-icon.png (180)             iOS home screen (opaque, square)
  * favicon-16.png / favicon-32.png        browser tab
  * favicon.ico                            legacy multi-res tab icon (16/32/48)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "design" / "icon-master.png"
OUT = ROOT / "public"
# The exact teal sampled from the master, used for maskable padding + the opaque
# apple-touch background so everything is seamless (also the manifest theme color).
TEAL = (9, 112, 108, 255)


def load_trimmed() -> Image.Image:
    """Master with the near-white outer corners made transparent, cropped tight to
    the rounded square. The glyph is separated from the corners by the teal border,
    so flood-filling from the corners never touches it."""
    image = Image.open(SRC).convert("RGBA")
    width, height = image.size
    for seed in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        ImageDraw.floodfill(image, seed, (0, 0, 0, 0), thresh=60)
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image


def resized(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.LANCZOS)


def on_teal(image: Image.Image, size: int, scale: float = 1.0) -> Image.Image:
    """Composite the glyph over a solid teal square (opaque). `scale` < 1 leaves a
    teal margin — used for maskable so the glyph sits inside the safe zone."""
    canvas = Image.new("RGBA", (size, size), TEAL)
    inner = round(size * scale)
    offset = (size - inner) // 2
    canvas.alpha_composite(image.resize((inner, inner), Image.LANCZOS), (offset, offset))
    return canvas.convert("RGB")


def main() -> None:
    trimmed = load_trimmed()

    # `purpose: any` + favicons — transparent corners, glyph edge-to-edge.
    resized(trimmed, 512).save(OUT / "icon-512.png")
    resized(trimmed, 192).save(OUT / "icon-192.png")
    resized(trimmed, 32).save(OUT / "favicon-32.png")
    resized(trimmed, 16).save(OUT / "favicon-16.png")

    # iOS home screen — opaque, square (iOS masks the corners itself).
    on_teal(trimmed, 180).save(OUT / "apple-touch-icon.png")

    # Android adaptive — teal to the edges, glyph within the 80% safe zone.
    on_teal(trimmed, 512, 0.82).save(OUT / "icon-512-maskable.png")
    on_teal(trimmed, 192, 0.82).save(OUT / "icon-192-maskable.png")

    # Legacy multi-resolution favicon.
    resized(trimmed, 256).save(
        OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)]
    )

    print("wrote icons to", OUT)


if __name__ == "__main__":
    main()
