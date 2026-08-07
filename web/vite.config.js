import { defineConfig } from "vite";

// COOP/COEP headers enable SharedArrayBuffer, which onnxruntime-web's
// threaded WASM backend needs. This only covers Vite's own dev/preview
// servers - a production static host needs the same two headers set at
// its own config level, or onnxruntime-web silently falls back to
// single-threaded WASM (still works, ~3-4x slower).
const crossOriginIsolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  // fixed, uncommon port - avoids colliding with other projects' dev
  // servers on the default 5173
  server: { port: 5299, headers: crossOriginIsolationHeaders },
  preview: { port: 5299, headers: crossOriginIsolationHeaders },
});
