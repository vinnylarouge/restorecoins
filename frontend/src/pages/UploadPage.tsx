import { useCallback, useState } from "react";
import { fileToBase64Png, base64ToDataUrl } from "../api/client";
import type { Action } from "../state/session";

export function UploadPage({ dispatch }: { dispatch: (a: Action) => void }) {
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    try {
      const b64 = await fileToBase64Png(file);
      dispatch({ type: "image_loaded", b64, dataUrl: base64ToDataUrl(b64) });
    } catch (e) {
      dispatch({ type: "error", message: `Could not read image: ${(e as Error).message}` });
    }
  }, [dispatch]);

  return (
    <div className="grid md:grid-cols-2 gap-8 items-start">
      <section>
        <h1 className="text-2xl mb-2">Restore a weathered coin</h1>
        <p className="text-ink/70 mb-4">
          Upload a single-coin photograph. The tool suggests where the coin appears
          weathered, you refine the mask, then it returns several candidate restorations
          alongside a sidecar JSON you can cite.
        </p>
        <ol className="list-decimal pl-5 space-y-1 text-sm text-ink/70">
          <li>Frontal view, neutral background, single coin in frame.</li>
          <li>Higher resolution helps — at least ~512px per side.</li>
          <li>Don't upload coins you can't legally share — images are not retained server-side, but you're sending them over the network.</li>
        </ol>
      </section>

      <section>
        <label
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
          className={`block panel cursor-pointer h-64 flex flex-col items-center justify-center text-center transition
                      ${dragOver ? "ring-2 ring-patina border-patina/40" : ""}`}
        >
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          />
          <span className="text-3xl mb-2">⤓</span>
          <span className="text-ink/70">Drop a coin photograph here, or click to choose.</span>
          <span className="text-xs text-ink/40 mt-1">PNG, JPG, WebP, TIFF accepted.</span>
        </label>
      </section>
    </div>
  );
}
