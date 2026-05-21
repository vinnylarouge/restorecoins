# BACKLOG.md

Everything explicitly out of scope for Phase A. See `PROJECT_SPEC.md §9`. Add to this file rather than smuggling things into Phase A.

## Deferred from spec §9

- [ ] Die-matching detection and die-matched paired training (Phase B).
- [ ] Type retrieval / Phase B coupling (CLIP-style OCRE retriever + IP-Adapter conditioning).
- [ ] Per-region segmentation / Phase C (replaces binary mask proposer; obverse-portrait, obverse-legend, reverse-iconography, reverse-legend, field, mint-mark labels).
- [ ] Per-pixel uncertainty (ensemble + region-specific calibration; UI hatching/transparency overlay).
- [ ] Legend OCR / legend completion.
- [ ] 3D restoration from multiple views.
- [ ] Mobile-first UI (responsive web is fine; native mobile is not).
- [ ] User accounts, saved restorations, history.
- [ ] Batch API for archaeological assemblages.
- [ ] Coin types outside Greek/Roman/Byzantine (Celtic, Islamic, Indian, Chinese).
- [ ] Multilingual UI.
- [ ] Paper writeup (Phase B/C deliverable).

## Discovered during Phase A (add as we go)

- [ ] **British Museum scraper.** OCRE→Fitzwilliam images are unreachable from some networks (see DECISIONS.md 2026-05-21). A BM scraper run from a cloud origin gives a redundant data path. Not Phase A because the SPARQL+image-download pattern is the same shape; only the endpoint differs.
- [ ] **IIIF-based image fetcher.** Many museums now expose IIIF endpoints (numismatics.org/ans/, fitzmuseum's IIIF service). IIIF supports server-side cropping/resize which would shrink the scrape footprint by ~10×. Phase A uses the lower-fidelity `foaf:depiction` URLs because they're zero-config; IIIF is a follow-on optimization.

<!-- format:
- [ ] Short title. Why it's not Phase A. Link to commit/issue that surfaced it.
-->
