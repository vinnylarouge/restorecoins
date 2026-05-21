"""Generate a fixture dataset of synthetic 'coin-like' images.

Used when the real OCRE scraper can't reach Fitzwilliam (firewall, offline,
rate-limited). Not training-quality data — the LoRA must never see these in
production. Purpose: exercise the dataloader, backend, and frontend end-to-end
without depending on network reachability.

A fixture coin is a disc with:
  - radial relief rings (Sobel-able structure for the exposure map)
  - a centred "portrait" blob (random ellipse)
  - a ring of "legend" marks around the rim (random short bars)
  - per-coin colour temperature jitter

Look subjectively coin-ish at thumbnail size. Do NOT use for evaluation;
classification recovery against fixtures is meaningless.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _coin(seed: int, size: int = 512) -> Image.Image:
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.ogrid[:h, :w]
    cy, cx, r = h // 2, w // 2, h // 2 - 8
    d = np.hypot(yy - cy, xx - cx)
    disc = (d < r).astype(np.float32)

    # Base coin colour: warmly metallic, with per-coin jitter.
    base_rgb = np.array([175 + rng.uniform(-30, 30),
                         145 + rng.uniform(-25, 25),
                         95 + rng.uniform(-20, 30)])
    rings = (np.cos(d * 0.6) * 0.5 + 0.5) * 50  # relief rings → gradient signal
    luma = disc * (120 + rings)

    rgb = np.stack([luma * base_rgb[i] / 175.0 for i in range(3)], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # Portrait ellipse, slightly off-centre, randomly oriented.
    p_w = int(r * rng.uniform(0.45, 0.65))
    p_h = int(r * rng.uniform(0.55, 0.75))
    pcx = cx + int(rng.uniform(-r * 0.05, r * 0.05))
    pcy = cy + int(rng.uniform(-r * 0.05, r * 0.05))
    p_colour = tuple(int(c) for c in base_rgb * rng.uniform(0.55, 0.75))
    draw.ellipse([pcx - p_w // 2, pcy - p_h // 2, pcx + p_w // 2, pcy + p_h // 2],
                 fill=p_colour)

    # "Legend" marks around the rim.
    n_marks = int(rng.integers(14, 26))
    for k in range(n_marks):
        theta = 2 * np.pi * k / n_marks + rng.uniform(-0.05, 0.05)
        rr = r - rng.integers(10, 22)
        mx = int(cx + rr * np.cos(theta))
        my = int(cy + rr * np.sin(theta))
        ms = int(rng.integers(3, 7))
        col = tuple(int(c) for c in base_rgb * rng.uniform(0.4, 0.7))
        draw.rectangle([mx - ms, my - ms, mx + ms, my + ms], fill=col)
    return img


def generate(out_dir: Path, n: int, size: int = 512, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Fake "types" — group fixtures into pretend type-series for eval scaffolding.
    n_types = max(1, n // 5)
    rows = [["coin_id", "coin_uri", "type_uri", "type_label",
             "obverse_path", "reverse_path"]]
    for i in range(n):
        type_idx = int(rng.integers(0, n_types))
        type_id = f"fixture-type-{type_idx:03d}"
        # Same per-type seed gives related-looking coins (shared portrait shape).
        coin_obv = _coin(seed * 1000 + type_idx * 100 + i * 2, size=size)
        coin_rev = _coin(seed * 1000 + type_idx * 100 + i * 2 + 1, size=size)
        obv_name = f"fixture_{i:05d}_obv.png"
        rev_name = f"fixture_{i:05d}_rev.png"
        coin_obv.save(out_dir / obv_name)
        coin_rev.save(out_dir / rev_name)
        rows.append([
            f"fixture_{i:05d}",
            f"https://example.org/fixtures/coin/{i:05d}",
            f"https://example.org/fixtures/type/{type_idx:03d}",
            f"Fixture Type {type_idx}",
            obv_name, rev_name,
        ])
    with (out_dir / "metadata.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    print(f"Generated {n} fixture coins ({n_types} types) at {out_dir}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a fixture coin dataset.")
    p.add_argument("--out", type=Path, default=Path("data/fixtures"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.out, n=args.n, size=args.size, seed=args.seed)


if __name__ == "__main__":
    _cli()
