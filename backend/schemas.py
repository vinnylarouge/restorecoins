"""Wire schemas that mirror PROJECT_SPEC.md §6 — binding across phases.

The TypeScript declarations on the frontend (`frontend/src/types/api.ts`) and
these Pydantic models are the same contract. Changes here require matched
changes there.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Conditioning(BaseModel):
    """Empty in Phase A. Populated in Phase B/C — see spec §5."""

    type_exemplars: Optional[list[str]] = None  # base64 PNGs of retrieved type exemplars
    region_labels: Optional[str] = None         # base64 PNG of per-pixel region labels


class RestoreRequest(BaseModel):
    image: str = Field(..., description="base64 PNG of the input coin image.")
    mask: str = Field(..., description="base64 PNG grayscale, 0=keep, 255=inpaint.")
    n_candidates: int = Field(4, ge=1, le=8)
    seed: Optional[int] = None
    conditioning: Optional[Conditioning] = None


class Provenance(BaseModel):
    model_id: str
    lora_id: str
    controlnet_id: Optional[str] = None
    seed: int
    sampler: str
    steps: int
    guidance_scale: float
    prompt: Optional[str] = None
    timestamp: str           # ISO 8601
    tool_version: str        # git SHA of backend
    input_image_sha256: str  # sidecar binds output to exact input bytes
    mask_sha256: str


class CandidateType(BaseModel):
    type_id: str             # Nomisma URI
    confidence: float
    exemplar_image_url: Optional[str] = None


class RestorationCandidate(BaseModel):
    image: str
    per_pixel_confidence: str  # uniform in Phase A; real map in Phase C
    provenance: Provenance
    candidate_types: Optional[list[CandidateType]] = None


class RestoreResponse(BaseModel):
    """Bundles the candidates so the frontend can iterate without extra parsing.

    A flat list would have matched the spec literally — bundling adds a tiny
    layer for future top-level fields (e.g., a server-side notice). The
    `RestorationCandidate` shape itself remains exactly per §6.
    """

    candidates: list[RestorationCandidate]


class ProposeMaskRequest(BaseModel):
    image: str


class ProposeMaskResponse(BaseModel):
    probability_map: str  # base64 PNG grayscale


class VersionResponse(BaseModel):
    tool_version: str
    model_id: str
    lora_id: str
    controlnet_id: Optional[str] = None
    mode: str
    schema_version: str
