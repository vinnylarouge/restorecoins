# Draft letter to a Heberden curator

Per PROJECT_SPEC §8 M10. Adapt names and any specific specimens; keep the tone deferential — they don't owe us their time.

---

> Subject: A web tool for restoring weathered ancient coins — would you try it and tell me what's wrong with it?
>
> Dear Dr. [Heuchert / Meadows / Howgego],
>
> I'm writing in the hope that you might spare ten minutes to try a web tool I've been building, and to tell me what it gets wrong.
>
> The tool is at **https://vinnylarouge.github.io/restorecoins/**. You upload a photograph of a weathered ancient coin; you paint or accept a mask over the weathered regions; the tool returns four candidate restorations, each accompanied by a sidecar JSON recording exactly the model, LoRA, seed, sampler, and image hashes used to produce it. The intent is purely diagnostic — to help narrow down candidate type attributions when the specimen is too worn to read confidently. The restorations are explicitly framed as hypotheses, not photographs; the UI watermarks every restored image to make that unmistakable.
>
> The architecture follows the line of work begun by Altaweel, Khelifi & Zafar (2024) and Baek & Choi (2025) but uses SDXL-inpainting with a coin-domain LoRA in place of the CycleGAN approaches those papers explored. The reproducibility-sidecar design is intended to make any restoration produced by the tool citable in academic work — a numismatist should be able to publish a restored image of a specimen with full provenance of the model that produced it.
>
> A few honest caveats:
> - The current LoRA was trained on ~2000 OCRE-typed Roman coins via Nomisma. It is heavily Phase A — biased toward common Roman imperial issues, untested on Greek, Celtic, or Byzantine corpora.
> - There is no type retrieval yet (that's Phase B). The tool restores; it does not identify.
> - The mask proposer is a small UNet trained on synthetic weathering; you may find its suggestions cosmetic and prefer to paint manually.
>
> What I would genuinely like to know is whether the tool, in its current state, would have any chance of being useful in your work — particularly for the long tail of corroded provincial bronzes that come through the Heberden as PAS finds or donations of unattributed assemblages. If the answer is "no, here's why" — that's the most valuable response I could receive, and I'd be enormously grateful for it.
>
> If you'd be willing to try the tool on one or two specimens you have to hand and write me a short reply (even just "useful" / "not useful" / "broken"), I would be in your debt.
>
> With thanks for your patience and your time,
>
> Vincent Wang-Maścianica
> vincentwangsemailaddress@gmail.com
> https://github.com/vinnylarouge/restorecoins

---

Anti-checklist before sending:

- [ ] Confirm the curator's current title and email. (`Volker Heuchert` was the Roman curator as of 2024; check `https://www.ashmolean.org/heberden-coin-room` for the current list.)
- [ ] Confirm the deployed URL actually loads and the backend responds in <60s.
- [ ] Test the tool yourself on a real weathered coin photo (not a fixture) and confirm at least one of the four candidates looks coherent.
- [ ] Confirm the sidecar JSON exports cleanly and the citation line in it is correct.
- [ ] Run the page through https://wave.webaim.org/ for accessibility — a curator on a screen reader should be able to use it.
- [ ] If your name doesn't match the institutional affiliation you want to claim, make that clear in the letter.

Things explicitly NOT to do in the first letter:

- Don't quote LPIPS or PSNR numbers (PROJECT_SPEC §2.1).
- Don't ask for a Heberden image-data partnership in the first contact (PROJECT_SPEC §4.1).
- Don't pitch this as a finished tool. It's a Phase A demo asking for guidance.
