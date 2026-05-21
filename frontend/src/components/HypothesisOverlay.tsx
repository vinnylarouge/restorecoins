// Spec §2(3): "Never hide that a restoration is a hypothesis. UI must always
// present multiple candidates and never display a single 'the answer' view.
// Visual indicators (e.g. subtle hatching overlay, watermark in metadata)
// should make it impossible to mistake a restoration for a photograph."
//
// We render a faint diagonal hatching pattern over restored images and stamp a
// small "hypothesis" label. The pattern is subtle enough not to obscure the
// coin, distinct enough that a screenshot still reads as a reconstruction.
export function HypothesisOverlay() {
  return (
    <>
      <div
        className="absolute inset-0 rounded pointer-events-none"
        style={{ backgroundImage: "var(--hatch-bg)" }}
        aria-hidden
      />
      <span
        className="absolute bottom-1 right-1 bg-parchment/85 text-ink/70
                   text-[10px] px-1.5 py-0.5 rounded font-mono pointer-events-none"
      >
        hypothesis
      </span>
    </>
  );
}
