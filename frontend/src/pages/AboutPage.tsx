import type { VersionResponse } from "../types/api";

export function AboutPage({
  onClose, version,
}: {
  onClose: () => void;
  version: VersionResponse | null;
}) {
  return (
    <article className="max-w-2xl mx-auto space-y-5">
      <header className="flex justify-between items-baseline">
        <h1 className="text-2xl">About · methodology · limitations</h1>
        <button className="btn" onClick={onClose}>← back</button>
      </header>

      <section className="space-y-2">
        <h2 className="text-lg">What this is</h2>
        <p className="text-ink/80">
          A diffusion-inpainting pipeline that proposes plausible restorations of
          weathered ancient coins. Its purpose is to <em>aid identification</em> —
          a numismatist looking at a corroded follis can use the restorations to
          narrow down between candidate types (Constantine vs. Licinius, for
          instance).
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">How it works</h2>
        <ol className="list-decimal pl-5 space-y-1 text-ink/80 text-sm">
          <li>You upload a coin photograph.</li>
          <li>A small UNet proposes which regions look weathered. You refine the mask by painting.</li>
          <li>An SDXL-inpainting model fine-tuned on a coin-domain LoRA proposes several candidate restorations. ControlNet-canny conditioning preserves surviving legend strokes.</li>
          <li>Each candidate ships with a sidecar JSON recording the model id, LoRA id, seed, sampler, and SHA256 hashes of the input. Anyone with the input can reproduce the exact restoration.</li>
        </ol>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Limitations</h2>
        <ul className="list-disc pl-5 space-y-1 text-ink/80 text-sm">
          <li><strong>A restoration is a hypothesis, not a photograph.</strong> The tool returns multiple candidates precisely because there is no single correct answer.</li>
          <li>Training data covers Greek/Roman/Byzantine coins. Celtic, Islamic, Indian, and Chinese coinage is out of scope for the current model.</li>
          <li>The mask proposer is heuristic in Phase A. A trained UNet improves it; you can override either way by painting.</li>
          <li>No type identification is performed. That's Phase B.</li>
          <li>Images are processed in-memory on the backend and discarded. Nothing is stored server-side, but they do travel over HTTPS to a HuggingFace Space.</li>
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Citation</h2>
        <p className="text-ink/80 text-sm">
          If you use a restoration in published work, please cite the tool version recorded in the sidecar JSON. The export modal generates a ready-to-paste citation string.
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Version</h2>
        {version ? (
          <dl className="font-mono text-xs bg-ink/5 p-3 rounded space-y-1">
            <FieldRow k="tool_version" v={version.tool_version} />
            <FieldRow k="model_id" v={version.model_id} />
            <FieldRow k="lora_id" v={version.lora_id} />
            <FieldRow k="controlnet_id" v={version.controlnet_id ?? "none"} />
            <FieldRow k="mode" v={version.mode} />
            <FieldRow k="schema_version" v={version.schema_version} />
          </dl>
        ) : <p className="text-ink/50 text-sm">Backend unreachable.</p>}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Acknowledgements</h2>
        <p className="text-ink/80 text-sm">
          Builds on OCRE (Online Coins of the Roman Empire) and the Nomisma linked-data stack. Methodologically indebted to Altaweel, Khelifi & Zafar (2024) and Baek & Choi (2025). Physical weathering inspired by Dorsey, Pedersen & Hanrahan (SIGGRAPH '96).
        </p>
      </section>
    </article>
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
