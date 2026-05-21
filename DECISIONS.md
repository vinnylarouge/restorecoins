# DECISIONS.md

Architectural and operational decisions, both resolved and open. Spec §10 lists the canonical open questions; this file is where their resolutions land.

## Resolved

### 2026-05-21 — Licensing
- **Code:** MIT (see `LICENSE`).
- **LoRA weights (when published):** CreativeML Open RAIL-M (matches SDXL upstream).
- **Documentation:** CC-BY-4.0 unless otherwise marked.

### 2026-05-21 — Hosting topology
- **Frontend:** GitHub Pages, served from `gh-pages` branch via workflow.
- **Backend:** HuggingFace Space (FastAPI on ZeroGPU), per spec §3.
- **Inference device:** `cuda` if available (ZeroGPU), `mps` on Apple Silicon for local dev, `cpu` fallback.

### 2026-05-21 — Python version
- Pinned to **3.12** via `.python-version`. Python 3.14 is too new for the diffusers/torch wheel ecosystem as of writing.

### 2026-05-21 — Primary inpainting model
- **SDXL-inpaint** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) + **ControlNet-canny SDXL** (`diffusers/controlnet-canny-sdxl-1.0`).
- FLUX.1 Fill comparison **deferred**: non-commercial licence (problem if a commercial spinoff ever happens) and ~24GB weights vs SDXL's ~7GB. The architecture (`pipeline.py`) accepts either model id, so swapping requires only a config change.

### 2026-05-21 — Backend mode flag
- `RESTORECOINS_MODE=mock` returns synthetically-degraded versions of the input as stand-in candidates. Used in CI, frontend dev, and dev machines without GPU. The §6 contract is identical in both modes — `mock` only changes how `RestorationCandidate.image` is generated.
- `RESTORECOINS_MODE=real` loads diffusion weights and runs SDXL-inpaint.

### 2026-05-21 — Data source reachability constraint (discovered during M1 smoke test)
- **Symptom:** All 500/500 sampled `foaf:depiction` URLs in Nomisma route to `www-img.fitzmuseum.cam.ac.uk`. From the development network the host is unreachable (TCP refused on 80/443; DNS resolves fine to `131.111.22.244`). British Museum's `collection.britishmuseum.org/sparql` times out from the same network; BM's REST API returns Cloudflare 403.
- **Implication:** The real scrape cannot complete from this network. The scraper itself works (SPARQL probe returns valid records).
- **Mitigation (Phase A):**
  1. `training/fixtures.py` generates a synthetic-coin dataset so the LoRA dataloader, backend `/restore`, and frontend can be smoke-tested without network.
  2. Real scraping deferred to a network with route to `www-img.fitzmuseum.cam.ac.uk` (Cambridge eduroam, Oxford VPN, or a cloud box).
  3. As a backup, add a British Museum scraper from a Cloudflare-allowed origin (e.g. a HuggingFace Space build) — tracked in BACKLOG.md.

### 2026-05-21 — Wikimedia Commons as the Phase A primary data source
- **Trigger:** Tested Oxford VPN — routes Oxford traffic but not Cambridge `.cam.ac.uk`, so the OCRE→Fitzwilliam path stays blocked. Wikimedia Commons (`upload.wikimedia.org`) is reachable from every network we've tried.
- **Decision:** Add `training/scrape_wikimedia.py` as the primary scrape source for Phase A. Walks the MediaWiki `categorymembers` API under "Roman coins by emperor" with depth-2 descent.
- **Trade-off:** Wikimedia type metadata is much weaker than OCRE (volunteer curation; no RIC numbers attached). For Phase A's LoRA the images themselves are what matters; for the Phase B type retriever we'll need OCRE-typed data and the Cambridge-routing problem must be solved.
- **Status of OCRE scraper:** kept as-is. When a Cambridge-routable network is available, `python -m training.scrape_ocre --limit 20000` is the one command to run.

### 2026-05-21 — MPS training defaults
- **Choice:** On Apple Silicon (`--device mps`), force fp32 (MPS fp16 has known NaN bugs in SDXL attention kernels), enable gradient checkpointing, and set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unimplemented ops.
- **Resolution:** Auto-tuning does *not* clamp `--resolution` on MPS — only warns. This honours explicit user intent (e.g. the spec calls for 1024²) at the cost of possible OOM on smaller Apple Silicon.
- **First M4 Max training run:** 1024², rank-32 LoRA, 2000 steps, fp32, batch=1, grad_accum=4, on a 27-image Wikimedia-Roman-coins corpus. Expected wall-clock ~16-24h. The 27-image corpus is small (spec aims at ~2000 pairs); the resulting LoRA will overfit and serve mainly as a proof of pipeline. Re-train with the full Wikimedia (or OCRE) corpus once the scrape is unblocked.

## Open (spec §10)

### Heberden contact
- **Status:** unverified.
- Spec names Volker Heuchert (Roman), Andrew Meadows / Chris Howgego (Greek) as last-known curators. **Action:** verify current lineup before reaching out for M10.

### Project naming
- **Working title:** "restorecoins" (repo), "Numismatic Deweatherer" (long-form in spec).
- Candidates: Restrike, Patina, Lustre, Heberden Restorer.
- **Action:** finalize before public launch. Ideally Heberden-blessed.

### GitHub org
- **Status:** undecided. Currently a personal repo on Vincent's account.
- Candidates: personal / Zefram AI Research / HAILab / new Oxford-flavoured org.
- **Affects:** deploy workflow `repository`, HF Space owner, LoRA Hub namespace.

### Hosting cost ceiling
- HF Spaces ZeroGPU is free for low traffic. **Action:** define monthly compute budget before tool is publicly cited.

### Training data redistribution
- Tool ships the trained LoRA, **not** the training images. OCRE images are mostly CC-BY or public domain but per-institution licensing varies.
- **Action:** confirm the "LoRA-as-derivative-work" position is defensible before publishing weights.
