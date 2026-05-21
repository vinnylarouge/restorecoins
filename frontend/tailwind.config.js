/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // A muted, museum-catalogue palette. Not pretty by design — the
        // restored image is the visual centrepiece, the chrome should recede.
        parchment: "#f4ede0",
        ink: "#1c1a14",
        patina: "#3a5a3a",
        bronze: "#8a6a3a",
        // Hypothesis hatching colour (see HypothesisOverlay).
        hatch: "rgba(28,26,20,0.18)",
      },
      fontFamily: {
        serif: ['"Source Serif Pro"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
