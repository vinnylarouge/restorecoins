"""Inpainting pipeline — mock for dev, real for SDXL-inpaint + LoRA + ControlNet.

The same `restore()` signature is honoured by both modes so the FastAPI handler
doesn't care which is loaded. `RESTORECOINS_MODE` env var picks between them.

Mock mode applies the synthetic-weathering library *in reverse* — a colour
shift and a mild deblur of the masked region — to produce plausible-looking
candidates that exercise the full response shape. It is NOT a real model and
will trivially fail eval; the only purpose is to make the frontend and the
contract testable without GPU.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter

from backend.provenance import build_provenance
from backend.schemas import RestorationCandidate

logger = logging.getLogger("restorecoins.pipeline")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

MODE = os.environ.get("RESTORECOINS_MODE", "mock").lower()
MODEL_ID = os.environ.get("RESTORECOINS_MODEL_ID",
                          "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_ID = os.environ.get("RESTORECOINS_CONTROLNET_ID",
                               "diffusers/controlnet-canny-sdxl-1.0")
# `none` means "no LoRA loaded" — the unmodified base model. Useful as a baseline.
LORA_ID = os.environ.get("RESTORECOINS_LORA_ID", "none")
DEFAULT_PROMPT = os.environ.get("RESTORECOINS_PROMPT",
                                "an ancient coin, museum photograph, sharp detail")
DEFAULT_NEG_PROMPT = "blurry, watermark, text overlay, modern coin, plastic"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Helpers shared by both modes                                                #
# --------------------------------------------------------------------------- #


def b64_to_pil(b64: str, mode: str = "RGB") -> tuple[Image.Image, bytes]:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert(mode)
    return img, raw


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def uniform_confidence(size: tuple[int, int], value: int = 128) -> str:
    """Phase A: a flat grey image stands in for the per-pixel confidence map.

    Spec §5: 'fill with a uniform value in Phase A but is a real map in Phase C'.
    Frontend renders it identically in either case so swapping in a real model
    is a backend-only change.
    """
    arr = np.full(size[::-1], value, dtype=np.uint8)
    return pil_to_b64(Image.fromarray(arr, mode="L"))


# --------------------------------------------------------------------------- #
# Mock pipeline                                                               #
# --------------------------------------------------------------------------- #


class MockPipeline:
    """Generates plausible-looking candidates without any diffusion model.

    Strategy per candidate:
      1. Lightly deblur the masked region (counteract the mechanical-wear blur).
      2. Reduce green tint in the masked region (counteract patination).
      3. Add per-candidate hue jitter so the N candidates differ visibly.

    None of this is a restoration — it's a contract exerciser.
    """

    name = "mock"

    def restore(
        self, image: Image.Image, mask: Image.Image,
        n_candidates: int, seed: int, prompt: str | None,
    ) -> Iterable[Image.Image]:
        rng = np.random.default_rng(seed)
        rgb = np.asarray(image).astype(np.float32)
        m = np.asarray(mask.resize(image.size).convert("L"), dtype=np.float32) / 255.0
        m3 = m[..., None]

        for i in range(n_candidates):
            # Deblur the masked region via unsharp-mask, then blend back by mask weight.
            unsharp = np.asarray(
                image.filter(ImageFilter.UnsharpMask(radius=1.5 + 0.3 * i, percent=160, threshold=2)),
                dtype=np.float32,
            )
            cand = rgb * (1 - m3) + unsharp * m3

            # Counter patination green: nudge green channel down in the masked region.
            green_nudge = np.zeros_like(cand)
            green_nudge[..., 1] = -15.0 * (1 + 0.2 * i)
            cand = cand + green_nudge * m3

            # Per-candidate hue jitter so candidates aren't identical.
            jitter = rng.uniform(-8, 8, size=3).astype(np.float32)
            cand = cand + jitter * m3
            cand = np.clip(cand, 0, 255).astype(np.uint8)
            yield Image.fromarray(cand)


# --------------------------------------------------------------------------- #
# Real pipeline (SDXL-inpaint + ControlNet-canny + optional LoRA)             #
# --------------------------------------------------------------------------- #


class RealPipeline:
    """Lazy-loads the SDXL-inpaint stack on first call.

    Loading is heavy (~10GB download on first run, ~30s reload from cache);
    we don't pay it at import time so the FastAPI app starts in mock mode if
    the user only wants to develop the UI.
    """

    name = "sdxl-inpaint+lora+controlnet"

    def __init__(self) -> None:
        self._pipe = None
        self._dev = device()

    def _load(self) -> None:
        if self._pipe is not None:
            return
        logger.info("Loading SDXL-inpaint + ControlNet-canny on %s", self._dev)
        from diffusers import (
            StableDiffusionXLControlNetInpaintPipeline,
            ControlNetModel,
            DPMSolverMultistepScheduler,
        )
        dtype = torch.float16 if self._dev in ("cuda", "mps") else torch.float32
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            MODEL_ID, controlnet=controlnet, torch_dtype=dtype,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if LORA_ID != "none":
            logger.info("Loading LoRA %s", LORA_ID)
            pipe.load_lora_weights(LORA_ID)
        pipe.to(self._dev)
        # On MPS the safety checker is a memory hog and irrelevant for coins.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        self._pipe = pipe

    def restore(
        self, image: Image.Image, mask: Image.Image,
        n_candidates: int, seed: int, prompt: str | None,
    ) -> Iterable[Image.Image]:
        self._load()
        prompt = prompt or DEFAULT_PROMPT
        # Use a fresh PRNG per candidate so seeds differ deterministically.
        for i in range(n_candidates):
            g = torch.Generator(device="cpu").manual_seed(seed + i)
            canny = _canny_control_image(image)
            out = self._pipe(
                prompt=prompt,
                negative_prompt=DEFAULT_NEG_PROMPT,
                image=image,
                mask_image=mask.convert("L"),
                control_image=canny,
                num_inference_steps=int(os.environ.get("RESTORECOINS_STEPS", "28")),
                guidance_scale=float(os.environ.get("RESTORECOINS_CFG", "7.0")),
                controlnet_conditioning_scale=float(
                    os.environ.get("RESTORECOINS_CN_SCALE", "0.6")
                ),
                strength=float(os.environ.get("RESTORECOINS_STRENGTH", "0.85")),
                generator=g,
            ).images[0]
            yield out


def _canny_control_image(image: Image.Image) -> Image.Image:
    """ControlNet-canny conditioning — preserves surviving legend strokes."""
    try:
        import cv2  # noqa: I001
        arr = np.asarray(image.convert("RGB"))
        edges = cv2.Canny(arr, 100, 200)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)
    except ImportError:
        # PIL fallback — coarser edges but doesn't add an OpenCV dependency.
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return Image.merge("RGB", [edges, edges, edges])


# --------------------------------------------------------------------------- #
# Public entrypoint                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    model_id: str
    lora_id: str
    controlnet_id: str | None
    sampler: str
    steps: int
    guidance_scale: float


def _config() -> PipelineConfig:
    is_real = MODE == "real"
    return PipelineConfig(
        model_id=MODEL_ID if is_real else "mock-deblur-v0",
        lora_id=LORA_ID if is_real else "none",
        controlnet_id=CONTROLNET_ID if is_real else None,
        sampler="DPMSolverMultistepScheduler" if is_real else "none",
        steps=int(os.environ.get("RESTORECOINS_STEPS", "28")) if is_real else 0,
        guidance_scale=float(os.environ.get("RESTORECOINS_CFG", "7.0")) if is_real else 0.0,
    )


# Module-global instances so model weights only load once.
_PIPELINE: MockPipeline | RealPipeline = MockPipeline() if MODE != "real" else RealPipeline()
_CONFIG = _config()


def get_pipeline_name() -> str:
    return _PIPELINE.name


def get_config() -> PipelineConfig:
    return _CONFIG


def restore(
    image_b64: str,
    mask_b64: str,
    n_candidates: int,
    seed: int | None,
) -> list[RestorationCandidate]:
    image, image_bytes = b64_to_pil(image_b64, "RGB")
    mask, mask_bytes = b64_to_pil(mask_b64, "L")
    seed = seed if seed is not None else int(np.random.SeedSequence().entropy & 0xFFFFFFFF)

    cfg = _CONFIG
    candidates: list[RestorationCandidate] = []
    confidence_b64 = uniform_confidence(image.size)
    for cand_img in _PIPELINE.restore(image, mask, n_candidates, seed, DEFAULT_PROMPT):
        prov = build_provenance(
            model_id=cfg.model_id,
            lora_id=cfg.lora_id,
            controlnet_id=cfg.controlnet_id,
            seed=seed,
            sampler=cfg.sampler,
            steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            prompt=DEFAULT_PROMPT if MODE == "real" else None,
            input_image_bytes=image_bytes,
            mask_bytes=mask_bytes,
        )
        candidates.append(RestorationCandidate(
            image=pil_to_b64(cand_img),
            per_pixel_confidence=confidence_b64,
            provenance=prov,
            candidate_types=None,  # Phase B/C
        ))
        seed += 1  # next candidate gets the next seed; sidecar records it
    return candidates


def propose_mask(image_b64: str) -> str:
    """Phase A: a placeholder mask proposer.

    Returns a base64 PNG grayscale where pixel intensity ≈ P(weathered). Until
    the M6 UNet is trained and shipped, we use a hand-rolled heuristic: low
    contrast + low saturation regions are more likely weathered. This is a
    deliberately stupid baseline so the frontend has something to render.
    """
    image, _ = b64_to_pil(image_b64, "RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # Local contrast = |pixel - local mean|, on a small window.
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(luma, size=11)
    contrast = np.abs(luma - local_mean)
    saturation = arr.max(axis=-1) - arr.min(axis=-1)
    weathered = 1.0 - (0.5 * (contrast / max(contrast.max(), 1e-6)) +
                       0.5 * (saturation / max(saturation.max(), 1e-6)))
    weathered = np.clip(weathered, 0, 1)
    out = (weathered * 255).astype(np.uint8)
    return pil_to_b64(Image.fromarray(out, mode="L"))
