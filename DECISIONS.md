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
