# restorecoins — Numismatic Deweatherer

A web tool that takes a photo of a weathered ancient coin and returns plausible restorations of how the coin looked closer to its minted appearance. Primary purpose: **aid identification** of weathered specimens against known type catalogues (RIC, RPC, SNG). Secondary: visualisation.

> **Status:** Phase A. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the binding spec, [BACKLOG.md](BACKLOG.md) for deferred items, and [DECISIONS.md](DECISIONS.md) for resolved/open architectural choices.

## What this is

A diffusion-inpainting pipeline (SDXL-inpaint + a coin-domain LoRA + ControlNet-canny) wrapped in a numismatist-facing UX with full reproducibility metadata. Every output ships with a sidecar JSON that records model identifiers, LoRA version, seed, sampler, mask, and input hash so a restored image can be cited in academic work.

```
┌──────────────────────┐  HTTPS+JSON   ┌────────────────────────────┐
│ GitHub Pages (static)│ ────────────▶ │ HuggingFace Space (FastAPI)│
│ React + Vite + TS    │ ◀──────────── │ SDXL-inpaint + LoRA + CN   │
└──────────────────────┘               └────────────────────────────┘
```

## Quick start (development)

If you have [`just`](https://github.com/casey/just):

```bash
just setup-py        # creates .venv, installs training+backend deps
just setup-fe        # npm install in frontend/
just fixtures        # generate 20 fake coins for offline smoke testing
just backend-mock &  # FastAPI on :7860, no GPU required
just dev-fe          # Vite dev server on :5173 → http://localhost:5173
```

Or manually:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt -r backend/requirements.txt
RESTORECOINS_MODE=mock uvicorn backend.app:app --reload --port 7860 &

cd frontend && npm install
VITE_BACKEND_URL=http://127.0.0.1:7860 VITE_BASE=/ npm run dev
```

For real-mode inference, `pip install -r backend/requirements-real.txt`, set `RESTORECOINS_MODE=real`, and ensure `diffusers`/`torch` can reach the model weights (first run downloads ~10GB).

## Repository layout

See [PROJECT_SPEC.md §7](PROJECT_SPEC.md). High level:

| Dir          | Purpose                                                  |
|--------------|----------------------------------------------------------|
| `frontend/`  | Vite + React + TS, deploys to GitHub Pages.              |
| `backend/`   | FastAPI app, deploys to HuggingFace Space (ZeroGPU).     |
| `training/`  | OCRE scraper, synthetic weathering, LoRA training, eval. |
| `data/`      | `.gitignored`. Scraped + synthetic pairs live here.      |
| `notebooks/` | Exploratory only — not the source of truth.              |

## Design principles (read [PROJECT_SPEC.md §2](PROJECT_SPEC.md) for the full list)

1. **Numismatists are the user, not ML researchers.** No FID scores in the UI.
2. **Reproducibility is required, not optional.** Sidecar JSON ships with every restoration.
3. **Never hide that a restoration is a hypothesis.** Multiple candidates, never one "answer".
4. **Design every interface for Phase C from day one.** §6 contract is binding across phases.

## License

MIT for code (this repo). Trained LoRA weights, when published, will be CreativeML Open RAIL-M to match SDXL upstream. Documentation is CC-BY-4.0 unless otherwise marked.

## Citation

If you use a restoration produced by this tool in published work, please cite the tool version recorded in the sidecar JSON. A `cite` button in the export modal generates a ready-to-paste citation.
