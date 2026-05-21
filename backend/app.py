"""FastAPI entrypoint for the Numismatic Deweatherer backend.

Endpoints (PROJECT_SPEC §4.7):
    POST /propose_mask   image (b64) → probability_map (b64 PNG grayscale)
    POST /restore        image, mask, N → list[RestorationCandidate]
    GET  /version        tool/model/lora ids + mode + schema version

Run locally:
    RESTORECOINS_MODE=mock uvicorn backend.app:app --reload --port 7860

For HF Spaces (ZeroGPU) deployment, see backend/Dockerfile and the README
frontmatter — the Space's runtime config picks up the same env vars.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import (
    MODE,
    LORA_ID,
    MODEL_ID,
    CONTROLNET_ID,
    propose_mask,
    restore,
)
from backend.provenance import SCHEMA_VERSION, tool_version
from backend.schemas import (
    ProposeMaskRequest,
    ProposeMaskResponse,
    RestoreRequest,
    RestoreResponse,
    VersionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restorecoins.app")

app = FastAPI(
    title="restorecoins",
    description="Numismatic Deweatherer backend — see PROJECT_SPEC.md.",
    version=SCHEMA_VERSION,
)

# §4.6/§4.7: frontend lives on GitHub Pages, backend on HF Spaces, different
# origins. CORS open in Phase A; revisit when rate-limiting is added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        tool_version=tool_version(),
        model_id=MODEL_ID if MODE == "real" else "mock-deblur-v0",
        lora_id=LORA_ID,
        controlnet_id=CONTROLNET_ID if MODE == "real" else None,
        mode=MODE,
        schema_version=SCHEMA_VERSION,
    )


@app.post("/propose_mask", response_model=ProposeMaskResponse)
def propose_mask_endpoint(req: ProposeMaskRequest) -> ProposeMaskResponse:
    try:
        prob_b64 = propose_mask(req.image)
    except Exception as e:
        logger.exception("propose_mask failed")
        raise HTTPException(status_code=400, detail=f"propose_mask failed: {e}")
    return ProposeMaskResponse(probability_map=prob_b64)


@app.post("/restore", response_model=RestoreResponse)
def restore_endpoint(req: RestoreRequest) -> RestoreResponse:
    try:
        candidates = restore(req.image, req.mask, req.n_candidates, req.seed)
    except Exception as e:
        logger.exception("restore failed")
        raise HTTPException(status_code=400, detail=f"restore failed: {e}")
    return RestoreResponse(candidates=candidates)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": MODE}


@app.get("/")
def root() -> dict:
    return {
        "name": "restorecoins",
        "description": "Numismatic Deweatherer backend.",
        "endpoints": ["/version", "/propose_mask", "/restore", "/health"],
        "spec": "PROJECT_SPEC.md §4.7",
    }
