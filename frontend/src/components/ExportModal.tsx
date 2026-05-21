import { useEffect } from "react";
import { downloadBase64, downloadJson } from "../api/client";
import type { SessionState } from "../state/session";
import type { VersionResponse } from "../types/api";

export function ExportModal({
  state, version, onClose,
}: {
  state: SessionState;
  version: VersionResponse | null;
  onClose: () => void;
}) {
  const selected = state.candidates[state.selectedCandidate];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const baseName = `restorecoin_${stamp}_seed${selected.provenance.seed}`;
  const sidecar = {
    schema_version: version?.schema_version ?? "unknown",
    provenance: selected.provenance,
    // Recording the candidate index lets you reproduce by re-running with the
    // same seed and asking for n_candidates ≥ index+1.
    candidate_index: state.selectedCandidate,
    n_candidates_total: state.candidates.length,
  };

  const citation = formatCitation(selected.provenance, version);

  return (
    <div
      className="fixed inset-0 bg-ink/40 flex items-center justify-center z-20 p-4"
      onClick={onClose}
    >
      <div
        className="bg-parchment rounded shadow-xl max-w-lg w-full p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between mb-3">
          <h3 className="font-serif text-lg">Export restoration</h3>
          <button className="btn text-xs" onClick={onClose}>esc</button>
        </div>

        <div className="space-y-3">
          <button
            className="btn btn-primary w-full justify-center"
            onClick={() => downloadBase64(selected.image, `${baseName}.png`)}
          >
            ↓ restored image (PNG)
          </button>
          <button
            className="btn w-full justify-center"
            onClick={() => downloadJson(sidecar, `${baseName}.sidecar.json`)}
          >
            ↓ reproducibility sidecar (JSON)
          </button>
          <details className="text-xs">
            <summary className="cursor-pointer text-ink/70 select-none">
              citation
            </summary>
            <pre className="mt-2 bg-ink/5 p-2 rounded font-mono text-[10px] whitespace-pre-wrap break-words">
              {citation}
            </pre>
            <button
              className="btn text-xs mt-2"
              onClick={() => navigator.clipboard.writeText(citation)}
            >copy</button>
          </details>
          <p className="text-[11px] text-ink/60">
            The sidecar contains the model id, LoRA id, seed, sampler, and SHA256
            hashes of the input image and mask. Anyone with the input can
            reproduce this exact restoration.
          </p>
        </div>
      </div>
    </div>
  );
}

function formatCitation(p: SessionState["candidates"][number]["provenance"], v: VersionResponse | null): string {
  const year = p.timestamp.slice(0, 4);
  const date = p.timestamp.slice(0, 10);
  return `Restored using restorecoins (${v?.tool_version.slice(0, 7) ?? "unknown"}), ${year}.
Model: ${p.model_id}; LoRA: ${p.lora_id}; ControlNet: ${p.controlnet_id ?? "none"}.
Seed: ${p.seed}; sampler: ${p.sampler}; steps: ${p.steps}; CFG: ${p.guidance_scale}.
Input SHA256: ${p.input_image_sha256}.
Restored on ${date}.`;
}
