import { useEffect, useState } from "react";
import { UploadPage } from "./pages/UploadPage";
import { MaskPage } from "./pages/MaskPage";
import { ResultsPage } from "./pages/ResultsPage";
import { AboutPage } from "./pages/AboutPage";
import { useSession } from "./state/session";
import { api } from "./api/client";
import type { VersionResponse } from "./types/api";

export default function App() {
  const [state, dispatch] = useSession();
  const [version, setVersion] = useState<VersionResponse | null>(null);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    api.version().then(setVersion).catch(() => setVersion(null));
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Header onAbout={() => setShowAbout(true)} version={version} />
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        {state.error && (
          <div className="panel border-red-300 bg-red-50 text-red-800 mb-4 flex justify-between">
            <span>⚠ {state.error}</span>
            <button className="btn" onClick={() => dispatch({ type: "clear_error" })}>
              dismiss
            </button>
          </div>
        )}
        {showAbout ? (
          <AboutPage onClose={() => setShowAbout(false)} version={version} />
        ) : state.stage === "upload" ? (
          <UploadPage dispatch={dispatch} />
        ) : state.stage === "mask" ? (
          <MaskPage state={state} dispatch={dispatch} />
        ) : (
          <ResultsPage state={state} dispatch={dispatch} version={version} />
        )}
      </main>
      <Footer />
    </div>
  );
}

function Header({ onAbout, version }: { onAbout: () => void; version: VersionResponse | null }) {
  return (
    <header className="border-b border-ink/10 bg-parchment/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <span className="font-serif text-xl">restorecoins</span>
          <span className="text-sm text-ink/60">Numismatic Deweatherer</span>
        </div>
        <div className="flex items-center gap-3">
          {version && (
            <span className="text-xs text-ink/50 font-mono">
              {version.mode}·{version.tool_version.slice(0, 7)}
            </span>
          )}
          <button className="btn text-sm" onClick={onAbout}>about · methodology</button>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="border-t border-ink/10 mt-8">
      <div className="max-w-6xl mx-auto px-4 py-4 text-xs text-ink/60 flex justify-between flex-wrap gap-2">
        <span>
          Every restoration is one hypothesis among many. Exported images include a sidecar JSON for reproducibility.
        </span>
        <span>
          MIT · <a className="underline" href="https://github.com/vinnylarouge/restorecoins">source</a>
        </span>
      </div>
    </footer>
  );
}
