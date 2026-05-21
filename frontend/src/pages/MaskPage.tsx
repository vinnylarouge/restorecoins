import { useEffect, useState } from "react";
import { MaskCanvas } from "../components/MaskCanvas";
import { api, base64ToDataUrl } from "../api/client";
import type { Action, SessionState } from "../state/session";

export function MaskPage({
  state, dispatch,
}: {
  state: SessionState;
  dispatch: (a: Action) => void;
}) {
  const [brushSize, setBrushSize] = useState(28);
  const [mode, setMode] = useState<"paint" | "erase">("paint");
  const [nCandidates, setNCandidates] = useState(4);
  const [threshold, setThreshold] = useState(140);

  // On first mount, ask the server for a suggested mask.
  useEffect(() => {
    if (!state.imageB64 || state.proposedMaskB64) return;
    dispatch({ type: "loading", what: "propose" });
    api.proposeMask(state.imageB64)
      .then((res) => dispatch({ type: "proposed_mask", b64: res.probability_map }))
      .catch((e) => dispatch({ type: "error", message: `propose_mask: ${e.message}` }))
      .finally(() => dispatch({ type: "loading", what: false }));
  }, [state.imageB64, state.proposedMaskB64, dispatch]);

  async function onRestore() {
    if (!state.imageB64 || !state.maskB64) {
      dispatch({ type: "error", message: "Paint a mask (or accept the suggested one) before restoring." });
      return;
    }
    dispatch({ type: "loading", what: "restore" });
    try {
      const res = await api.restore({
        image: state.imageB64,
        mask: state.maskB64,
        n_candidates: nCandidates,
      });
      dispatch({ type: "candidates", candidates: res.candidates });
    } catch (e) {
      dispatch({ type: "error", message: `restore: ${(e as Error).message}` });
    }
  }

  return (
    <div className="grid md:grid-cols-[1fr_280px] gap-6">
      <section className="panel">
        <h2 className="font-serif text-lg mb-3">Refine the inpainting mask</h2>
        {state.imageDataUrl && (
          <MaskCanvas
            imageDataUrl={state.imageDataUrl}
            proposedMaskB64={state.proposedMaskB64}
            threshold={threshold}
            brushSize={brushSize}
            mode={mode}
            onMaskChange={(b64) =>
              dispatch({ type: "user_mask", b64, dataUrl: base64ToDataUrl(b64) })
            }
          />
        )}
        <p className="text-xs text-ink/60 mt-2">
          The shaded region will be inpainted. Painted regions adopt the new
          texture; unpainted regions are kept as-is. Areas with surviving legend
          strokes or iconography are best left unpainted.
        </p>
      </section>

      <aside className="space-y-4">
        <div className="panel">
          <h3 className="font-serif text-base mb-2">Mask tools</h3>
          <div className="flex gap-2 mb-2">
            <button
              className={`btn flex-1 ${mode === "paint" ? "btn-primary" : ""}`}
              onClick={() => setMode("paint")}
            >paint</button>
            <button
              className={`btn flex-1 ${mode === "erase" ? "btn-primary" : ""}`}
              onClick={() => setMode("erase")}
            >erase</button>
          </div>
          <label className="block text-xs text-ink/70 mb-1">brush · {brushSize}px</label>
          <input type="range" min={4} max={120} value={brushSize}
                 onChange={(e) => setBrushSize(+e.target.value)} className="w-full" />
          <label className="block text-xs text-ink/70 mt-3 mb-1">suggestion threshold · {threshold}</label>
          <input type="range" min={0} max={255} value={threshold}
                 onChange={(e) => setThreshold(+e.target.value)} className="w-full" />
          <p className="text-[10px] text-ink/50 mt-1">
            Lower threshold → more pixels are marked weathered by default.
          </p>
        </div>

        <div className="panel">
          <h3 className="font-serif text-base mb-2">Restoration</h3>
          <label className="block text-xs text-ink/70 mb-1">candidates · {nCandidates}</label>
          <input type="range" min={1} max={8} value={nCandidates}
                 onChange={(e) => setNCandidates(+e.target.value)} className="w-full" />
          <button
            className="btn btn-primary w-full justify-center mt-3"
            disabled={state.loading !== false}
            onClick={onRestore}
          >
            {state.loading === "restore" ? "restoring…"
             : state.loading === "propose" ? "preparing suggestion…"
             : "restore"}
          </button>
          <button
            className="btn w-full justify-center mt-2 text-xs"
            onClick={() => dispatch({ type: "back_to_upload" })}
          >
            ← upload a different coin
          </button>
        </div>
      </aside>
    </div>
  );
}
