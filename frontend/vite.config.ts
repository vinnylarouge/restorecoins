import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// `base` matters for GitHub Pages: served at /<repo>/ unless on a custom domain.
// Override at build time with `VITE_BASE=/` if the org switches to a CNAME.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  return {
    plugins: [react()],
    base: env.VITE_BASE ?? "/restorecoins/",
    server: { port: 5173, proxy: {
      // Convenience proxy so `fetch('/api/...')` works in dev without CORS.
      "/api": {
        target: env.VITE_BACKEND_URL ?? "http://localhost:7860",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    } },
  };
});
