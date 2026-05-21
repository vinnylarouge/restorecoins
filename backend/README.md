---
title: restorecoins backend
emoji: "🪙"
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Diffusion-inpainting restoration of weathered ancient coins.
license: mit
---

# restorecoins backend

FastAPI + SDXL-inpaint + ControlNet-canny + a coin-domain LoRA, packaged for HuggingFace Spaces (Docker SDK, ZeroGPU runtime).

The frontmatter above is what HF Spaces uses for repo config — do not strip it. Move the README in the repo root takes precedence over this one for human reading; this file's job is the YAML.

## Endpoints

| Method | Path             | Purpose                                                     |
|--------|------------------|-------------------------------------------------------------|
| `POST` | `/restore`        | Generate N candidate restorations. See `schemas.py` for the request/response shape. |
| `POST` | `/propose_mask`   | Suggest a weathered-region probability map.                  |
| `GET`  | `/version`        | Model/LoRA versions + mode + schema version (used by sidecar). |
| `GET`  | `/health`         | Liveness check.                                              |

## Run locally

```bash
# Mock mode (no GPU, no model downloads):
RESTORECOINS_MODE=mock uvicorn backend.app:app --reload --port 7860

# Real mode (downloads SDXL-inpaint + ControlNet-canny on first call, ~10GB):
pip install -r backend/requirements-real.txt
RESTORECOINS_MODE=real \
RESTORECOINS_LORA_ID=vinnylarouge/restorecoins-lora-v0.1 \
uvicorn backend.app:app --reload --port 7860
```

## Environment variables

| Var                                  | Default                                                       | Notes                                            |
|--------------------------------------|---------------------------------------------------------------|--------------------------------------------------|
| `RESTORECOINS_MODE`                  | `mock`                                                        | `mock` or `real`.                                |
| `RESTORECOINS_MODEL_ID`              | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`            | HF id of the base inpainting model.              |
| `RESTORECOINS_CONTROLNET_ID`         | `diffusers/controlnet-canny-sdxl-1.0`                         | HF id of the ControlNet.                         |
| `RESTORECOINS_LORA_ID`               | `none`                                                        | `none` or an HF model id of a LoRA.              |
| `RESTORECOINS_PROMPT`                | `"an ancient coin, museum photograph, sharp detail"`          | Neutral domain prompt; see `pipeline.py`.        |
| `RESTORECOINS_STEPS`                 | `28`                                                          | Sampler steps.                                   |
| `RESTORECOINS_CFG`                   | `7.0`                                                         | Classifier-free guidance scale.                  |
| `RESTORECOINS_CN_SCALE`              | `0.6`                                                         | ControlNet conditioning scale.                   |
| `RESTORECOINS_STRENGTH`              | `0.85`                                                        | Inpainting strength.                             |
| `RESTORECOINS_MASK_PROPOSER_PATH`    | (unset)                                                       | Path to M6 UNet checkpoint; falls back to heuristic. |
