"""Inference wrapper around the trained mask-proposer UNet (M6).

Loaded lazily — if no checkpoint is present, the backend's
`pipeline.propose_mask` falls back to its heuristic. Keeping this module
optional lets the backend start without M6 training having finished.

Once a checkpoint exists at `RESTORECOINS_MASK_PROPOSER_PATH`, set the env var
and the backend will use the learned proposer instead of the heuristic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.train_mask_proposer import MaskProposer

logger = logging.getLogger("restorecoins.mask_proposer")

_CHECKPOINT_PATH = os.environ.get("RESTORECOINS_MASK_PROPOSER_PATH", "")
_MODEL: MaskProposer | None = None
_DEVICE: str | None = None


def _ensure_loaded() -> MaskProposer | None:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    if not _CHECKPOINT_PATH or not Path(_CHECKPOINT_PATH).exists():
        return None
    _DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Loading mask proposer from %s onto %s", _CHECKPOINT_PATH, _DEVICE)
    # weights_only=False because we save a dict that includes the training
    # config (with PosixPath) alongside state_dict. We trust our own checkpoint.
    ckpt = torch.load(_CHECKPOINT_PATH, map_location=_DEVICE, weights_only=False)
    model = MaskProposer().to(_DEVICE).eval()
    model.load_state_dict(ckpt["state_dict"])
    _MODEL = model
    return _MODEL


def propose(image: Image.Image, resolution: int = 512) -> np.ndarray:
    """Return a probability map (H×W, float in [0, 1]) for the input image.

    Falls back to None (caller must handle) if no checkpoint is available; the
    backend's heuristic stays the default until a real proposer is trained.
    """
    model = _ensure_loaded()
    if model is None:
        raise FileNotFoundError(
            f"Mask proposer checkpoint not found at {_CHECKPOINT_PATH!r}. "
            "Either train via `python -m training.train_mask_proposer` and set "
            "RESTORECOINS_MASK_PROPOSER_PATH, or rely on the heuristic in "
            "backend.pipeline.propose_mask."
        )
    img = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    # Resize back to original size for compositing in the UI.
    prob_pil = Image.fromarray((prob * 255).astype(np.uint8), mode="L").resize(image.size)
    return np.asarray(prob_pil, dtype=np.float32) / 255.0
