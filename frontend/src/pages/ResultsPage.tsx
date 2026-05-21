import { useState } from "react";
import { base64ToDataUrl } from "../api/client";
import { ExportModal } from "../components/ExportModal";
import { HypothesisOverlay } from "../components/HypothesisOverlay";
import type { Action, SessionState } from "../state/session";
import type { VersionResponse } from "../types/api";

export function ResultsPage({
  state, dispatch, version,
}: {
  state: SessionState;
  dispatch: (a: Action) => void;
  version: VersionResponse | null;
}) {
  const [compareMode, setCompareMode] = useState<"side" | "toggle">("side");
  const [showOriginal, setShowOriginal] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  const selected = state.candidates[state.selectedCandidate];
  if (!selected || !state.imageDataUrl) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-serif text-xl">Candidate restorations</h2>
          <p className="text-xs text-ink/60">
            {state.candidates.length} hypotheses. None is a photograph — every restoration
            is one possible reconstruction. Sidecar JSON ships with the export.
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn" onClick={() => dispatch({ type: "back_to_mask" })}>
            ← refine mask
          </button>
          <button className="btn btn-primary" onClick={() => setExportOpen(true)}>
            export ↓
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-[1fr_220px] gap-4">
        <section className="panel">
          <div className="flex items-center justify-between mb-3 text-sm">
            <div className="flex gap-2">
              <button
                className={`btn ${compareMode === "side" ? "btn-primary" : ""}`}
                onClick={() => setCompareMode("side")}
              >side-by-side</button>
              <button
                className={`btn ${compareMode === "toggle" ? "btn-primary" : ""}`}
                onClick={() => setCompareMode("toggle")}
              >toggle</button>
            </div>
            <span className="text-xs text-ink/50 font-mono">
              cand {state.selectedCandidate + 1}/{state.candidates.length}
              · seed={selected.provenance.seed}
            </span>
          </div>

          {compareMode === "side" ? (
            <div className="grid grid-cols-2 gap-3">
              <ImageCard label="weathered input" src={state.imageDataUrl} />
              <ImageCard label="restored hypothesis"
                         src={base64ToDataUrl(selected.image)} hypothesis />
            </div>
          ) : (
            <div className="space-y-3">
              <div
                onMouseDown={() => setShowOriginal(true)}
                onMouseUp={() => setShowOriginal(false)}
                onMouseLeave={() => setShowOriginal(false)}
                onTouchStart={() => setShowOriginal(true)}
                onTouchEnd={() => setShowOriginal(false)}
                className="cursor-pointer select-none"
              >
                <ImageCard
                  label={showOriginal ? "weathered input (hold)" : "restored hypothesis (release)"}
                  src={showOriginal ? state.imageDataUrl : base64ToDataUrl(selected.image)}
                  hypothesis={!showOriginal}
                />
              </div>
              <p className="text-[11px] text-ink/50 text-center">
                Hold pointer to peek at the original.
              </p>
            </div>
          )}
        </section>

        <aside className="space-y-3">
          <div className="panel">
            <h3 className="font-serif text-base mb-2">All candidates</h3>
            <div className="grid grid-cols-2 gap-2">
              {state.candidates.map((c, i) => (
                <button
                  key={i}
                  onClick={() => dispatch({ type: "select_candidate", index: i })}
                  className={`relative aspect-square overflow-hidden rounded border
                              ${i === state.selectedCandidate
                                ? "border-patina ring-1 ring-patina"
                                : "border-ink/15 hover:border-ink/40"}`}
                  title={`seed ${c.provenance.seed}`}
                >
                  <img src={base64ToDataUrl(c.image)} className="w-full h-full object-cover" />
                  <span className="absolute top-1 left-1 bg-parchment/80 text-[10px] px-1 rounded font-mono">
                    {i + 1}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="panel text-xs text-ink/70">
            <h3 className="font-serif text-base text-ink mb-2">Provenance · selected</h3>
            <dl className="space-y-1 font-mono text-[11px]">
              <FieldRow k="model" v={selected.provenance.model_id} />
              <FieldRow k="lora" v={selected.provenance.lora_id} />
              {selected.provenance.controlnet_id && (
                <FieldRow k="cnet" v={selected.provenance.controlnet_id} />
              )}
              <FieldRow k="seed" v={String(selected.provenance.seed)} />
              <FieldRow k="sampler" v={selected.provenance.sampler} />
              <FieldRow k="steps" v={String(selected.provenance.steps)} />
              <FieldRow k="cfg" v={selected.provenance.guidance_scale.toFixed(2)} />
              <FieldRow k="input" v={selected.provenance.input_image_sha256.slice(0, 12) + "…"} />
              <FieldRow k="time" v={selected.provenance.timestamp} />
              <FieldRow k="tool" v={selected.provenance.tool_version.slice(0, 12)} />
            </dl>
          </div>
        </aside>
      </div>

      {exportOpen && (
        <ExportModal
          state={state}
          version={version}
          onClose={() => setExportOpen(false)}
        />
      )}
    </div>
  );
}

function ImageCard({ label, src, hypothesis }: { label: string; src: string; hypothesis?: boolean }) {
  return (
    <figure className="relative">
      <img src={src} className="w-full rounded border border-ink/10" alt={label} />
      {hypothesis && <HypothesisOverlay />}
      <figcaption className="text-[11px] text-ink/60 mt-1">{label}</figcaption>
    </figure>
  );
}

function FieldRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-ink/50">{k}</dt>
      <dd className="truncate" title={v}>{v}</dd>
    </div>
  );
}
