"""Dataset adapters shared across train_lora.py and train_mask_proposer.py.

The training pairs are generated on-the-fly from clean OCRE images via the
synthetic weathering library. We deliberately do *not* cache pairs to disk:
caching freezes the severity distribution at the moment of caching, while
on-the-fly sampling exposes the LoRA to fresh weatherings every step and
generalises better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.synthetic_weathering import WeatherParams, weather


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def list_clean_coins(root: Path) -> list[Path]:
    """Recursively list image files under `root`, sorted for determinism."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


class WeatheredCoinPairs(Dataset):
    """Yields `(weathered_tensor, pristine_tensor, mask_tensor)` triples.

    All tensors are float32 in [0, 1], CHW. Mask is single-channel.
    """

    def __init__(
        self,
        root: Path,
        resolution: int = 1024,
        severity_low: float = 0.2,
        severity_high: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.paths = list_clean_coins(root)
        if not self.paths:
            raise ValueError(f"No images under {root}; can't build training set.")
        self.resolution = resolution
        self.severity_low = severity_low
        self.severity_high = severity_high
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        pristine = Image.open(path).convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        # Per-sample seed: same image + same severity-draw is reproducible,
        # but two epochs over the same image draw different weatherings.
        sample_seed = int(self.rng.integers(0, 2**31 - 1))
        sev = self.rng.uniform(self.severity_low, self.severity_high, size=3)
        params = WeatherParams(
            mechanical=float(sev[0]), corrosion=float(sev[1]), patination=float(sev[2]),
            seed=sample_seed,
        )
        weathered_arr, mask_arr = weather(pristine, params)

        pristine_t = _to_chw(np.asarray(pristine))
        weathered_t = _to_chw(weathered_arr)
        mask_t = torch.from_numpy(mask_arr.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "weathered": weathered_t,
            "pristine": pristine_t,
            "mask": mask_t,
            "path": str(path),
        }


def _to_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    arr = rgb_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
