# Numismatic Deweatherer — Project Specification

**Version:** 0.1 (Phase A scope, Phase C-aware architecture)
**Audience:** Implementation engineer (Claude Code or otherwise)
**Status:** Greenlit, building Phase A

---

## 1. Project summary

A web tool for numismatists that takes a photo of a weathered ancient coin and returns plausible restorations of how the coin looked closer to its minted appearance. The primary purpose is to **aid identification** of weathered specimens against known type catalogues (RIC, RPC, SNG, etc.); the secondary purpose is visualisation.

This project resurrects an older grant from the Oxford Archaeology department to build numismatic identification software. The original implementation thesis — train deweatherers on synthetically weathered coins — is still valid, but the model landscape has moved on to the point where open-weight diffusion inpainting models do most of the heavy lifting. Modern scope: smart fine-tuning, smart conditioning, smart UX, smart evaluation. Not from-scratch model architecture.

**Phase A target (this spec):** A working web demo, frontend on GitHub Pages, backend on HuggingFace Spaces, that a curator at the Ashmolean's Heberden Coin Room could use today. SDXL-inpaint or FLUX.1 Fill with a coin-specific LoRA, user-painted or auto-proposed masks, N candidate restorations, full reproducibility sidecar.

**Phase C aspiration (architect for, do not build):** Type-coupled probabilistic restoration with calibrated per-region uncertainty, integrated with OCRE/Nomisma retrieval. Every interface in Phase A must be designed to accommodate this without rewriting.

---

## 2. Design principles (read these before making any decision)

These are non-negotiable. When in doubt, re-read.

1. **Numismatists are the user, not ML researchers.** No FID scores in the UI. No "model confidence: 0.847". The user wants to know whether they're looking at a Constantine follis or a Licinius follis. Everything in the UX should serve that decision.

2. **Reproducibility is required, not optional.** Every restoration produced by the tool must be exportable with a sidecar JSON containing: model identifier and version, LoRA identifier and version, mask used, seed, sampler settings, prompt (if any), input image hash. A numismatist cannot publish a restored image without being able to say exactly how it was produced. Without this, the tool is useless for academic work.

3. **Never hide that a restoration is a hypothesis.** The output is *one possible reconstruction*, not *the* reconstruction. UI must always present multiple candidates and never display a single "the answer" view. Visual indicators (e.g. subtle hatching overlay, watermark in metadata) should make it impossible to mistake a restoration for a photograph.

4. **Design every interface for Phase C from day one.** Interface contracts in §6 are the binding spec. Implementations can be trivial in Phase A but signatures must not change.

5. **Don't reinvent what exists.** Use the Altaweel OCRE scraper, use diffusers, use the standard fine-tuning pipelines. Build the parts where there is genuine novelty (coin-domain LoRA, weathered-region detector, numismatist UX, reproducibility layer).

6. **Ship a thing a Heberden curator can actually click on within 4 weeks.** Resist scope creep. Every "wouldn't it be nice if" goes in `BACKLOG.md`, not in Phase A.

---

## 3. Architecture overview

```
                                  ┌─────────────────────────────────────┐
                                  │  GitHub Pages (static frontend)     │
                                  │  - React + Vite + TypeScript        │
                                  │  - Mask painting (Konva or similar) │
                                  │  - Candidate gallery UI             │
                                  │  - Sidecar export                   │
                                  └──────────────┬──────────────────────┘
                                                 │ HTTPS, JSON+base64
                                                 ▼
                                  ┌─────────────────────────────────────┐
                                  │  HuggingFace Space (Gradio/FastAPI) │
                                  │  - SDXL-inpaint + coin LoRA         │
                                  │  - ControlNet-canny conditioning    │
                                  │  - Mask proposer (small UNet)       │
                                  │  - Reproducibility sidecar builder  │
                                  │  - ZeroGPU runtime                  │
                                  └─────────────────────────────────────┘
```

**Why this split:** GitHub Pages cannot run models. HF Spaces with ZeroGPU is free for research use, handles SDXL-class models, and the API contract is just HTTPS+JSON. Frontend and backend are decoupled — backend can move to Modal/Replicate/self-hosted later without touching the frontend.

**Why not in-browser inference (WebGPU + ONNX):** SDXL is ~3-5GB to ship to the client and 30-60s per inference on a good GPU. Numismatists will use this on laptops, often on conference Wi-Fi. Server-side wins.

**Why not Gradio for the whole thing:** Gradio UX ceiling is too low for the reproducibility and multi-candidate workflows we need. Use Gradio's serving capability (via `gradio.mount_gradio_app` or just FastAPI on a Space) but build the actual UI in React.

---

## 4. Phase A scope — what we are building

### 4.1 Data sources

In priority order:

1. **OCRE (Online Coins of the Roman Empire) via Nomisma SPARQL.**
   - Endpoint: `http://nomisma.org/query`
   - Use the scraper from Altaweel, Khelifi & Zafar 2024 (JCAA, supplementary material): https://journal.caa-international.org/articles/10.5334/jcaa.146 — check the article's supplementary files for the scraper code. If unavailable or insufficient, write a fresh SPARQL-based scraper.
   - Target: ~10,000–20,000 high-grade specimens with type metadata.
   - Filter aggressively for image quality: minimum resolution, frontal view, single-coin images, neutral background. Most online coin images are auction photos — quality is highly variable.

2. **British Museum API.** https://www.britishmuseum.org/our-work/departments/coins-and-medals — has an API; permissive licensing for non-commercial research use. Use as supplementary training data and as evaluation data (their weathered finds are a real-world degradation distribution).

3. **Portable Antiquities Scheme (PAS).** https://finds.org.uk/database — has a long tail of weathered finds. Use **only for evaluation**, not training — these are the kind of inputs the tool will see in the wild.

4. **Heberden Coin Room.** Defer until institutional conversation has happened. If/when available, becomes the primary high-grade source and the project gains a real institutional partner. **Do not block Phase A on this.**

### 4.2 Synthetic weathering pipeline

Two implementations, both required:

**(a) Naive corruption (baseline, ~half a day):**
- Gaussian noise, motion blur, random patch masking, low-pass filtering, brightness/contrast perturbation, JPEG compression artifacts.
- This is the bad baseline. Train a model only on (a) to show it overfits the synthetic distribution.

**(b) Physically-motivated weathering (the real training signal, ~1 week):**
- Treat the coin image as an approximation of a height map (gradient magnitude as a proxy for relief height; this is crude but works for low-relief coins).
- Compute an **exposure map** (high points of relief, derived from gradient magnitude and local maxima) and a **recess map** (low points, derived from local minima of brightness in inverted lighting).
- Simulate three degradation modes:
  - **Mechanical wear**: smooth high-exposure regions preferentially (Gaussian blur weighted by exposure map).
  - **Corrosion/pitting**: add localised dark spots with size distribution drawn from observations of real corroded coins (parameterise from PAS images).
  - **Patination/encrustation**: add greenish-grey colour shifts weighted by recess map, with spatially correlated noise.
- Each mode parameterised by severity (0-1). Sample uniformly during training.
- Reference: Dorsey, Pedersen & Hanrahan "Modeling and Rendering of Weathered Stone" (SIGGRAPH 1996) and follow-ups for the physical intuition, but the implementation here is a simple 2D approximation, not a full BRDF model.

**(c) Die-matched pairs (DEFERRED to Phase B):** Real paired training data from die-matched specimens of different conditions. Powerful but requires solving die-matching. Out of scope for Phase A.

### 4.3 Model

**Primary:** SDXL-inpaint (`stabilityai/stable-diffusion-xl-base-1.0` + `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) with:
- A **coin-domain LoRA** fine-tuned on high-grade OCRE coins. Target rank 32-64, ~1000-2000 training steps, trained on the synthetic-weathering forward model (weathered input → pristine target).
- **ControlNet-canny** conditioning on edge maps of the input image, to preserve surviving legend strokes and iconography during inpainting.

**Comparison:** FLUX.1 Fill [dev] (`black-forest-labs/FLUX.1-Fill-dev`) with the same LoRA approach if feasible. Note FLUX licence is non-commercial — fine for research and the Heberden, but flag if a commercial spinoff is ever envisaged.

**Mask proposer (auto-suggest):** Small UNet trained to predict "weathered vs. preserved" probability per pixel. Training data: synthetic weathering process gives perfect ground-truth masks for free. Inference: outputs a probability map; UI converts to a binary mask at a user-adjustable threshold; user can paint over to refine.

**Do not:**
- Train a model from scratch (Phase D).
- Use CycleGAN (the 2024 papers establish it underperforms diffusion).
- Use single-image super-resolution as a substitute (different problem).

### 4.4 Training pipeline

- Framework: `diffusers` + `peft` (HuggingFace stack). Use `accelerate` for multi-GPU if available, single-GPU is fine for the LoRA.
- LoRA training: ~6-12 hours on a single A100 or equivalent. Vincent's existing setup should handle this; if not, use a Lambda or RunPod box for the training run.
- Save LoRA weights to HuggingFace Hub under a project org. **Tag versions semantically** — every deployed model must be referenceable by an immutable tag for the reproducibility sidecar.

### 4.5 Evaluation

Three layers:

1. **Pixel/perceptual metrics on synthetic test set.** PSNR, SSIM, LPIPS on held-out synthetically-weathered images. These are diagnostic, not the goal.

2. **Type-classification recovery rate.** Take real well-preserved coins from OCRE with known type, apply synthetic weathering, restore, then run a pretrained coin classifier (or CLIP retrieval against OCRE type exemplars) on (weathered) and (restored). Restoration is useful iff top-k classification accuracy improves. **This is the operational metric for Phase A.**

3. **Numismatist evaluation (Phase A.5, post-launch):** Designed pilot study with 3-5 Heberden curators / visiting scholars. Blind comparison: weathered image only vs. weathered + restored. Measure correct attribution rate, time-to-attribution, and self-reported confidence. This is what determines whether the tool *actually works* in the sense that matters.

### 4.6 Frontend

- **Stack:** Vite + React + TypeScript + Tailwind. Konva.js for mask painting on canvas.
- **Pages:**
  - **Upload page:** drag-drop image, see preview, "next" button.
  - **Mask page:** image with auto-proposed mask overlay; user can paint to add/remove mask regions; brush size, opacity controls; "lock" mode for regions to preserve.
  - **Results page:** grid of N candidate restorations (default N=4); each candidate shows original/restored toggle; click any candidate to enlarge with side-by-side comparison.
  - **Export modal:** download restored image as PNG/TIFF; downloads the sidecar JSON alongside; "copy citation" button generates a citation string referencing the tool version.
- **No login, no accounts, no analytics beyond aggregate page hits.** Numismatists are a small, privacy-conscious community; the tool's job is to be useful, not to capture users.
- **Stateless frontend:** all state lives in URL params or browser session; no backend database in Phase A.

### 4.7 Backend (HuggingFace Space)

- **Stack:** FastAPI on a ZeroGPU Space. Use Gradio's auth-token mechanism for rate limiting if abuse becomes a problem; in Phase A, open access.
- **Endpoints:**
  - `POST /propose_mask` — input: image (base64). Output: probability map (base64 PNG, grayscale).
  - `POST /restore` — input: image (base64), mask (base64), N candidates, optional seed. Output: list of `RestorationCandidate` objects (see §6).
  - `GET /version` — returns model + LoRA + code versions for sidecar.
- **No data retention:** images received are processed in-memory and discarded. State this prominently in the UI.

---

## 5. Phase B/C forward-compatibility — what to design for now

These are not built in Phase A but the interfaces in §6 must accommodate them.

**Phase B additions:**
- Type retriever: CLIP-style image encoder trained on OCRE, returns top-k candidate types for a query image.
- Retrieved exemplars become additional conditioning to the inpainter (via IP-Adapter or FLUX Redux).
- Output adds a "candidate types" list alongside the restoration candidates.

**Phase C additions:**
- Region segmenter replaces binary mask proposer: outputs per-pixel labels (obverse portrait, obverse legend, reverse iconography, reverse legend, field, mint mark, weathered, background).
- Per-region inpainting with region-specific models or conditioning.
- Per-pixel uncertainty: ensemble over seeds + region-specific calibration. Output is a confidence map, not a single image.
- UI shows the restored image with uncertainty overlay (hatching, transparency, or a toggle).

**The implication for Phase A:** the `RestorationCandidate` schema (§6) has a `per_pixel_confidence` field that is filled with a uniform value in Phase A but is a real map in Phase C. Build the UI to handle the field even if it's degenerate now.

---

## 6. Interface contracts (binding)

These are the shapes that must not change between phases.

```typescript
// Sent to backend
interface RestoreRequest {
  image: string;              // base64 PNG
  mask: string;               // base64 PNG, grayscale, 0=keep, 255=inpaint
  n_candidates: number;       // default 4, max 8
  seed?: number;              // optional, for reproducibility
  conditioning?: {            // empty in Phase A, populated in B/C
    type_exemplars?: string[];      // base64 images of retrieved type exemplars
    region_labels?: string;         // base64 PNG, per-pixel region label map
  };
}

// Returned from backend
interface RestorationCandidate {
  image: string;              // base64 PNG of restored coin
  per_pixel_confidence: string; // base64 PNG grayscale, uniform value in Phase A
  provenance: {
    model_id: string;         // e.g. "sdxl-inpaint-1.0"
    lora_id: string;          // e.g. "ox-numismatics/coin-lora-v0.3"
    controlnet_id?: string;
    seed: number;
    sampler: string;
    steps: number;
    guidance_scale: number;
    prompt?: string;
    timestamp: string;        // ISO 8601
    tool_version: string;     // git commit SHA of backend
  };
  candidate_types?: Array<{   // empty in Phase A, populated in B/C
    type_id: string;          // Nomisma URI
    confidence: number;
    exemplar_image_url?: string;
  }>;
}
```

The sidecar JSON exported with each restoration is exactly the `provenance` field plus a hash of the input image and mask.

---

## 7. Repository layout

```
numismatic-deweatherer/
├── README.md
├── PROJECT_SPEC.md          # this file
├── BACKLOG.md               # all "wouldn't it be nice if" items
├── frontend/                # Vite + React + TS, deploys to GitHub Pages
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── backend/                 # FastAPI app, deploys to HF Space
│   ├── app.py
│   ├── pipeline.py          # the inpainting orchestration
│   ├── mask_proposer.py
│   ├── provenance.py
│   ├── requirements.txt
│   └── Dockerfile           # for Space build
├── training/                # LoRA training, mask-proposer training
│   ├── scrape_ocre.py
│   ├── synthetic_weathering.py
│   ├── train_lora.py
│   ├── train_mask_proposer.py
│   └── evaluate.py
├── data/                    # NOT committed; .gitignored
│   ├── raw/                 # scraped images
│   ├── filtered/            # after quality filtering
│   └── synthetic_pairs/     # (weathered, pristine) training pairs
├── notebooks/               # exploratory work, not the source of truth
└── .github/
    └── workflows/
        ├── deploy-frontend.yml   # auto-deploy to gh-pages
        └── deploy-backend.yml    # auto-push to HF Space
```

---

## 8. Build order (Phase A milestones)

Each milestone is a green-light gate. Do not start the next one without the previous one working.

**M1 — Data corpus (week 1, ~3 days).**
- Scrape OCRE; end state: `data/raw/` with ≥10,000 images and a metadata CSV.
- Quality filter; end state: `data/filtered/` with ≥5,000 high-grade images.
- Acceptance: a notebook displays a random grid of 16 filtered images and they all look like clean, high-grade coins.

**M2 — Synthetic weathering (week 1, ~2 days).**
- Implement naive corruption (a).
- Implement physically-motivated weathering (b).
- End state: `training/synthetic_weathering.py` is a library that takes a clean coin image and outputs a weathered version + ground-truth mask.
- Acceptance: a notebook shows 8 originals + 8 weathered versions side-by-side, and the weathered ones look subjectively like real corroded coins (not just noisy clean coins).

**M3 — Baseline inpainting (week 2, ~2 days).**
- Stand up SDXL-inpaint locally via diffusers.
- Run on 10 hand-selected weathered coins with hand-painted masks, save the results.
- Acceptance: results are visibly worse than a coin domain expert would accept. This establishes the floor.

**M4 — LoRA fine-tune (week 2, ~3 days).**
- Train coin-domain LoRA on ~2000 (synthetically weathered, pristine) pairs.
- Acceptance: re-run M3's 10 examples with the LoRA loaded; outputs are visibly better. Upload LoRA to HF Hub with a version tag.

**M5 — ControlNet integration (week 3, ~2 days).**
- Add ControlNet-canny conditioning.
- Acceptance: on a test image where the legend is partially preserved, inpainted output preserves the surviving legend strokes (no hallucinated overwrites). This is a hard test; iterate on conditioning strength until it passes.

**M6 — Mask proposer (week 3, ~2 days).**
- Small UNet trained on synthetic weathering ground-truth masks.
- Acceptance: on 20 real weathered coins, proposed mask reasonably identifies weathered regions (sanity-check by eye, not a metric).

**M7 — Backend (week 3-4, ~3 days).**
- FastAPI app with the three endpoints.
- Deploy to HF Space.
- Acceptance: a `curl` to the deployed `/restore` endpoint returns a valid response within 60s.

**M8 — Frontend (week 4, ~4 days).**
- Vite+React app with upload, mask, results, export pages.
- Deploy to GitHub Pages.
- Acceptance: an end-to-end run by a non-developer (ask Paulina, ask a Heberden curator) succeeds without instructions.

**M9 — Reproducibility & polish (week 4-5, ~3 days).**
- Sidecar JSON export.
- Citation generator.
- "About this tool" page with methodology, limitations, citation request.
- Acceptance: a restoration export contains everything needed to reproduce the exact same output on the backend.

**M10 — First user contact (week 5).**
- Email Heberden curator with a demo link and a short writeup.
- Acceptance: numismatist replies with feedback (positive or negative — either is success).

Total: ~5 weeks of focused work. Realistic for one person working part-time.

---

## 9. What is explicitly out of scope for Phase A

Add to `BACKLOG.md`, do not implement:

- Die-matching detection and die-matched paired training.
- Type retrieval / Phase B coupling.
- Per-region segmentation / Phase C.
- Legend OCR / legend completion.
- 3D restoration from multiple views.
- Mobile-first UI (web-responsive is fine; native mobile is not).
- User accounts, saved restorations, history.
- Batch processing / API for archaeological assemblages.
- Coin types outside Greek/Roman/Byzantine (Celtic, Islamic, Indian etc.) — focus on the largest training corpus first.
- Multilingual UI.
- An actual paper. The paper is the Phase B/C deliverable; Phase A is the tool.

---

## 10. Open questions to resolve before / during M1

These should be answered in the first week, with stub decisions documented in `DECISIONS.md`:

1. **Heberden contact.** Confirm current curator(s) — Volker Heuchert (Roman, last known), Andrew Meadows / Chris Howgego (Greek)? Verify the current lineup before reaching out.
2. **Project naming.** "Numismatic Deweatherer" is the working title. Something better before public launch (suggested: "Restrike", "Patina", "Lustre", "Heberden Restorer"). Ideally Heberden-blessed.
3. **GitHub org.** Personal repo, Zefram AI Research, HAILab, or a new Oxford-flavoured org?
4. **Licensing.** Code: MIT or Apache-2.0. Model weights (LoRA): CreativeML Open RAIL-M to match SDXL upstream. Documentation: CC-BY-4.0. Confirm.
5. **Hosting cost.** HF Spaces ZeroGPU is free for low traffic. If the tool gets cited and traffic grows, monthly cost ceiling? (Pro tier is ~$9/month for the user but Spaces compute scales independently.)
6. **Image rights.** OCRE images are mostly CC-BY or public domain but per-institution licensing varies. The tool should not redistribute training data; only the trained LoRA. Confirm the LoRA-as-derivative-work position is defensible.

---

## 11. Reference reading (do not need to read all; cited for context)

- Altaweel, Khelifi & Zafar (2024). "Using Generative AI for Reconstructing Cultural Artifacts: Examples Using Roman Coins." *JCAA* 7(1): 301-315. — direct predecessor, uses CycleGAN.
- Zachariou, Dimitriou & Arandjelović (2020). "Visual reconstruction of ancient coins using cycle-consistent generative adversarial networks." *Sci.* 2(3): 52. — first formulation.
- Baek & Choi (2025). "RICH: Coin data restoration using inpainting for cultural heritage analysis." *ETRI Journal.* — first diffusion-based formulation, on Korean coins.
- Dorsey, Pedersen & Hanrahan (1996). "Modeling and Rendering of Weathered Stone." *SIGGRAPH '96.* — for the synthetic weathering physical intuition.
- Schlag & Arandjelović (2017). "Ancient Roman coin recognition in the wild using deep learning based recognition of artistically depicted face profiles." *ICCV-W.* — for the downstream classification task.
- OCRE: http://numismatics.org/ocre/
- Nomisma SPARQL: http://nomisma.org/query
- FLUX.1 Fill [dev]: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
- SDXL inpaint: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1

---

## 12. Handoff notes for Claude Code

- Read this file in full before starting.
- Create `BACKLOG.md` and `DECISIONS.md` as empty files at repo init; populate as you go.
- Treat §6 (interface contracts) and §2 (design principles) as non-negotiable; flag for Vincent before deviating.
- Each milestone (§8) ends with a commit message of the form `feat(MN): <description>` so progress is greppable.
- When in doubt about a numismatic detail (what counts as a "high-grade" image, what regions of a coin are most informative, what counts as a "successful restoration"), flag for Vincent rather than guess — there is no good way to derive this from first principles.
- Do not commit anything to `data/`. Add `data/` to `.gitignore` at repo init.
- Backend deployment to HF Spaces uses git; create the Space and add it as a remote.
- Frontend deployment to GitHub Pages uses the `gh-pages` branch via the workflow in `.github/workflows/deploy-frontend.yml`.
