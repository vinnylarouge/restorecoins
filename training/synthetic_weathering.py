"""Synthetic weathering for the LoRA training pipeline.

Implements PROJECT_SPEC.md §4.2:
    (a) Naive corruption — the baseline that overfits to its own noise distribution.
    (b) Physically-motivated weathering — exposure/recess maps drive three degradation
        modes (mechanical wear, corrosion/pitting, patination/encrustation).

Both produce `(weathered_image, ground_truth_mask)`. The mask records which pixels
were significantly altered: it is the training target for the mask-proposer UNet
(§4.3) and the inpainting region for the LoRA fine-tune (§4.4).

References:
    Dorsey, Pedersen & Hanrahan, "Modeling and Rendering of Weathered Stone",
    SIGGRAPH '96, for the physical intuition. This module is a 2D approximation,
    not a full BRDF model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

WeatherMode = Literal["mechanical", "corrosion", "patination"]
ALL_MODES: tuple[WeatherMode, ...] = ("mechanical", "corrosion", "patination")

# Pixels whose colour distance from the original exceeds this (out of ~441 max
# for sRGB) are considered "weathered" for the ground-truth mask. Tuned to track
# the visual threshold at which a region looks materially altered.
_MASK_THRESHOLD_DELTA = 18.0


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class WeatherParams:
    """Severity in [0, 1] per mode, plus a master seed for reproducibility."""

    mechanical: float = 0.5
    corrosion: float = 0.5
    patination: float = 0.5
    seed: int | None = None
    modes: tuple[WeatherMode, ...] = field(default_factory=lambda: ALL_MODES)


def weather(
    image: Image.Image | np.ndarray,
    params: WeatherParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply physically-motivated weathering. Returns (rgb_uint8, mask_uint8).

    `mask_uint8` is 255 where the pixel was significantly altered, 0 elsewhere.
    Suitable as both the LoRA inpainting region and the mask-proposer target.
    """
    params = params or WeatherParams()
    rgb = _as_rgb_array(image)
    rng = np.random.default_rng(params.seed)

    exposure = _exposure_map(rgb)
    recess = _recess_map(rgb)

    out = rgb.astype(np.float32)
    if "mechanical" in params.modes and params.mechanical > 0:
        out = _mechanical_wear(out, exposure, params.mechanical, rng)
    if "corrosion" in params.modes and params.corrosion > 0:
        out = _corrosion_pitting(out, params.corrosion, rng)
    if "patination" in params.modes and params.patination > 0:
        out = _patination(out, recess, params.patination, rng)

    out = np.clip(out, 0, 255).astype(np.uint8)
    mask = _delta_mask(rgb, out)
    return out, mask


def naive_corruption(
    image: Image.Image | np.ndarray,
    severity: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spec §4.2(a): the baseline that overfits its own synthetic distribution.

    Composes Gaussian noise, motion blur, random patch masking, low-pass
    filtering, brightness/contrast perturbation, and JPEG artifacts. Useful as
    a foil: a LoRA trained only on this should fail visibly on real weathered
    coins. If it succeeds, the eval is the bug, not the model.
    """
    rgb = _as_rgb_array(image)
    rng = np.random.default_rng(seed)
    s = float(np.clip(severity, 0, 1))

    out = rgb.astype(np.float32)
    out = out + rng.normal(0, 12 * s, size=out.shape)
    out = _motion_blur(out, kernel_size=int(3 + 8 * s))
    out = _low_pass(out, sigma=0.5 + 1.5 * s)
    out = _brightness_contrast(out, brightness=1.0 + rng.uniform(-0.2, 0.2) * s,
                               contrast=1.0 + rng.uniform(-0.2, 0.2) * s)
    out = _random_patch_mask(out, n_patches=int(2 + 8 * s),
                             max_frac=0.05 + 0.10 * s, rng=rng)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _jpeg_artifacts(out, quality=int(85 - 60 * s))
    mask = _delta_mask(rgb, out)
    return out, mask


# --------------------------------------------------------------------------- #
# Physically-motivated maps                                                   #
# --------------------------------------------------------------------------- #


def _exposure_map(rgb: np.ndarray) -> np.ndarray:
    """High points of relief, in [0, 1].

    Gradient magnitude (Sobel) of luminance is a crude proxy for relief slope;
    smoothed and normalized, peaks identify exposed ridges — exactly the
    surfaces that get mechanically worn first on a coin in circulation.
    """
    luma = _luminance(rgb)
    gx = ndimage.sobel(luma, axis=1)
    gy = ndimage.sobel(luma, axis=0)
    grad = np.hypot(gx, gy)
    grad = ndimage.gaussian_filter(grad, sigma=2.0)
    return _normalize01(grad)


def _recess_map(rgb: np.ndarray) -> np.ndarray:
    """Low points of relief, in [0, 1].

    Inverted luminance + low-pass approximates the recessed regions where
    patination preferentially accumulates (mineral salts pool in lows, edges
    self-clean as the coin rubs against pocket/soil).
    """
    luma = _luminance(rgb)
    inv = 255.0 - luma
    inv = ndimage.gaussian_filter(inv, sigma=3.0)
    return _normalize01(inv)


# --------------------------------------------------------------------------- #
# Degradation modes                                                           #
# --------------------------------------------------------------------------- #


def _mechanical_wear(
    rgb: np.ndarray, exposure: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Blur weighted by the exposure map, so ridges smear before fields do."""
    s = float(np.clip(severity, 0, 1))
    sigma = 0.8 + 4.0 * s
    blurred = np.stack(
        [ndimage.gaussian_filter(rgb[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    # Per-pixel mix weight: high exposure → blurred, low exposure → original.
    w = (exposure ** (1.5 - 0.5 * s))[..., None]
    return blurred * w + rgb * (1.0 - w)


def _corrosion_pitting(
    rgb: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Stamp dark, irregular pits whose sizes follow a heavy-tailed power law.

    Spec calls for a size distribution "parameterised from PAS images"; until we
    have those, a power-law with exponent ~2.5 is a reasonable proxy — most pits
    are small, a few are large, none are perfectly round.
    """
    s = float(np.clip(severity, 0, 1))
    h, w = rgb.shape[:2]
    out = rgb.copy()
    n_pits = int(20 + 200 * s)
    diag = float(np.hypot(h, w))

    for _ in range(n_pits):
        # Power-law radius in pixels: r ∝ U^(-1/(alpha-1)), alpha=2.5.
        r_norm = max(0.001, rng.random()) ** (-1.0 / 1.5)
        radius = float(np.clip(r_norm * 0.5, 1.0, 0.04 * diag))
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        _stamp_pit(out, cy, cx, radius, severity=s, rng=rng)
    return out


def _stamp_pit(
    rgb: np.ndarray, cy: int, cx: int, radius: float,
    severity: float, rng: np.random.Generator,
) -> None:
    """In-place: paint a soft, irregular dark blob at (cy, cx)."""
    h, w = rgb.shape[:2]
    r = int(np.ceil(radius * 1.8))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(yy - cy, xx - cx)
    # Smooth falloff + a sliver of high-frequency noise → irregular edges.
    falloff = np.clip(1.0 - d / max(1.0, radius), 0, 1)
    falloff = falloff ** (1.5 + rng.uniform(-0.3, 0.3))
    jitter = rng.normal(0, 0.08, size=falloff.shape)
    alpha = np.clip(falloff + jitter * falloff, 0, 1) * (0.55 + 0.35 * severity)
    # Pit colour: darker than local mean, with a slight warm/cool jitter.
    local = rgb[y0:y1, x0:x1].mean(axis=(0, 1))
    pit_colour = local * rng.uniform(0.15, 0.40) + rng.uniform(-8, 8, size=3)
    blend = alpha[..., None]
    rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - blend) + pit_colour * blend


def _patination(
    rgb: np.ndarray, recess: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Greenish-grey shift weighted by the recess map; spatially correlated."""
    s = float(np.clip(severity, 0, 1))
    h, w = rgb.shape[:2]

    # Spatially-correlated noise via low-pass-filtered white noise; gives the
    # blotchy, organic look real patina has versus uniform tinting.
    correlated = ndimage.gaussian_filter(rng.standard_normal((h, w)),
                                         sigma=8.0)
    correlated = _normalize01(correlated)

    weight = (recess * correlated)[..., None]
    weight = np.clip(weight * (0.4 + 0.6 * s), 0, 0.85)

    # Bronze disease green: roughly (90, 120, 90), with per-pixel jitter.
    target = np.array([90.0, 120.0, 90.0]) + rng.uniform(-15, 15, size=3)
    target = np.broadcast_to(target, rgb.shape).copy()
    target += rng.normal(0, 10, size=rgb.shape)

    return rgb * (1.0 - weight) + target * weight


# --------------------------------------------------------------------------- #
# Naive-corruption primitives                                                 #
# --------------------------------------------------------------------------- #


def _motion_blur(rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, kernel_size | 1)  # force odd
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return np.stack(
        [ndimage.convolve(rgb[..., c], kernel, mode="reflect") for c in range(3)],
        axis=-1,
    )


def _low_pass(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack(
        [ndimage.gaussian_filter(rgb[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )


def _brightness_contrast(rgb: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    mean = rgb.mean()
    return (rgb - mean) * contrast + mean * brightness


def _random_patch_mask(
    rgb: np.ndarray, n_patches: int, max_frac: float, rng: np.random.Generator
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.copy()
    for _ in range(n_patches):
        ph = int(rng.integers(4, max(5, int(h * max_frac))))
        pw = int(rng.integers(4, max(5, int(w * max_frac))))
        y = int(rng.integers(0, h - ph))
        x = int(rng.integers(0, w - pw))
        fill = rng.uniform(0, 255, size=3)
        out[y:y + ph, x:x + pw] = fill
    return out


def _jpeg_artifacts(rgb: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 5, 95))
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr[..., :3]
    return np.asarray(image.convert("RGB"))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _delta_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Pixels whose L2 colour distance exceeds the threshold get marked 255."""
    delta = np.linalg.norm(before.astype(np.float32) - after.astype(np.float32), axis=-1)
    return (delta > _MASK_THRESHOLD_DELTA).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# CLI: stamp a small demo grid so M2's acceptance criterion is self-checkable #
# --------------------------------------------------------------------------- #


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate a weathered-vs-original demo grid.")
    p.add_argument("input", type=Path, help="Path to a clean coin image (or a directory of them).")
    p.add_argument("output", type=Path, help="Output PNG path.")
    p.add_argument("--n", type=int, default=8, help="Number of samples in the grid.")
    p.add_argument("--mode", choices=["physical", "naive"], default="physical")
    p.add_argument("--severity", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = sorted(args.input.glob("*")) if args.input.is_dir() else [args.input]
    paths = [pp for pp in paths if pp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not paths:
        raise SystemExit(f"No images found at {args.input}")

    rng = np.random.default_rng(args.seed)
    chosen = list(rng.choice(paths, size=min(args.n, len(paths)), replace=False))
    rows = []
    for pth in chosen:
        img = Image.open(pth).convert("RGB").resize((256, 256))
        if args.mode == "physical":
            weathered, _ = weather(img, WeatherParams(
                mechanical=args.severity, corrosion=args.severity,
                patination=args.severity, seed=args.seed,
            ))
        else:
            weathered, _ = naive_corruption(img, severity=args.severity, seed=args.seed)
        pair = np.concatenate([np.asarray(img), weathered], axis=1)
        rows.append(pair)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(args.output)
    print(f"Wrote {args.output} ({len(chosen)} pairs).")


if __name__ == "__main__":
    _cli()
