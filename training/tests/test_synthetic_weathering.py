"""Sanity tests for the synthetic weathering library.

Not a quality check (that's the M2 visual acceptance grid). These tests guard
the API contract — shape, dtype, mask coverage — so the LoRA trainer never gets
silently malformed pairs.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from training.synthetic_weathering import (
    ALL_MODES,
    WeatherParams,
    naive_corruption,
    weather,
)


@pytest.fixture
def clean_coin() -> Image.Image:
    """A coin-shaped synthetic stand-in: bright disc with radial 'relief' rings."""
    h = w = 128
    yy, xx = np.ogrid[:h, :w]
    cy, cx, r = h // 2, w // 2, h // 2 - 4
    d = np.hypot(yy - cy, xx - cx)
    disc = (d < r).astype(np.float32)
    # Rings give the gradient/Sobel something to find, exercising the exposure map.
    rings = (np.cos(d * 0.5) * 0.5 + 0.5) * 60
    base = disc * (140 + rings)
    rgb = np.stack([base, base * 0.95, base * 0.85], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def test_weather_returns_rgb_uint8_and_mask(clean_coin: Image.Image) -> None:
    weathered, mask = weather(clean_coin, WeatherParams(seed=42))
    assert weathered.shape == (128, 128, 3)
    assert weathered.dtype == np.uint8
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_weather_actually_alters_image(clean_coin: Image.Image) -> None:
    original = np.asarray(clean_coin)
    weathered, mask = weather(clean_coin, WeatherParams(seed=42))
    assert not np.array_equal(original, weathered)
    # Mask should mark at least 1% of pixels — otherwise the threshold is wrong
    # and the LoRA trainer would see "weathered" pairs with empty inpainting regions.
    assert mask.mean() > 2.55  # ~1% of 255


def test_weather_is_deterministic_with_seed(clean_coin: Image.Image) -> None:
    a, ma = weather(clean_coin, WeatherParams(seed=7))
    b, mb = weather(clean_coin, WeatherParams(seed=7))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma, mb)


def test_severity_zero_is_close_to_identity(clean_coin: Image.Image) -> None:
    weathered, mask = weather(
        clean_coin,
        WeatherParams(mechanical=0, corrosion=0, patination=0, seed=0),
    )
    np.testing.assert_array_equal(weathered, np.asarray(clean_coin))
    assert mask.sum() == 0


@pytest.mark.parametrize("mode", ALL_MODES)
def test_single_mode_produces_change(clean_coin: Image.Image, mode: str) -> None:
    p = WeatherParams(mechanical=0, corrosion=0, patination=0, seed=1, modes=(mode,))
    setattr(p, mode, 0.8)
    weathered, mask = weather(clean_coin, p)
    assert not np.array_equal(weathered, np.asarray(clean_coin))
    assert mask.sum() > 0


def test_naive_corruption_contract(clean_coin: Image.Image) -> None:
    weathered, mask = naive_corruption(clean_coin, severity=0.5, seed=3)
    assert weathered.shape == (128, 128, 3)
    assert weathered.dtype == np.uint8
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8


def test_accepts_ndarray_input(clean_coin: Image.Image) -> None:
    arr = np.asarray(clean_coin)
    weathered, mask = weather(arr, WeatherParams(seed=0))
    assert weathered.shape == arr.shape
    assert mask.shape == arr.shape[:2]
