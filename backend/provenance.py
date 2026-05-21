"""Sidecar / provenance helpers (PROJECT_SPEC §2(2), §4.7).

A restored image is academically useless without a record of how it was
produced. This module produces that record as a Provenance object that ships
with every RestorationCandidate. The frontend writes it out as a sidecar JSON
file at export time.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache

from backend.schemas import Provenance

SCHEMA_VERSION = "1.0"


def sha256_b64(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@lru_cache(maxsize=1)
def tool_version() -> str:
    """Git SHA of the backend, falling back to env var, then 'unknown'.

    On HF Spaces the build sets `SPACE_COMMIT` or `GIT_COMMIT_HASH`; locally we
    shell out to git. Caching is fine because tool_version doesn't change at
    runtime within a single process.
    """
    for env_var in ("SPACE_COMMIT", "GIT_COMMIT_HASH", "GITHUB_SHA"):
        if (v := os.environ.get(env_var)):
            return v
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=3, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def build_provenance(
    *,
    model_id: str,
    lora_id: str,
    controlnet_id: str | None,
    seed: int,
    sampler: str,
    steps: int,
    guidance_scale: float,
    prompt: str | None,
    input_image_bytes: bytes,
    mask_bytes: bytes,
) -> Provenance:
    return Provenance(
        model_id=model_id,
        lora_id=lora_id,
        controlnet_id=controlnet_id,
        seed=seed,
        sampler=sampler,
        steps=steps,
        guidance_scale=guidance_scale,
        prompt=prompt,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        tool_version=tool_version(),
        input_image_sha256=sha256_b64(input_image_bytes),
        mask_sha256=sha256_b64(mask_bytes),
    )
