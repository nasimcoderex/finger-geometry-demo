// Copies model assets from the Python project and MediaPipe's WASM runtime
// from node_modules into public/, so everything is self-hosted (no CDN
// dependency, no CORS/rate-limit risk, works offline once cached).
// Run via `predev`/`prebuild` npm script hooks - see package.json.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const pyRoot = dirname(webRoot);
const publicModels = join(webRoot, "public", "models");
const publicMediapipeWasm = join(webRoot, "public", "mediapipe-wasm");

mkdirSync(publicModels, { recursive: true });

const modelAssets = [
  [join(pyRoot, "model.onnx"), join(publicModels, "model.onnx")],
  [join(pyRoot, "models", "hand_landmarker.task"), join(publicModels, "hand_landmarker.task")],
  [join(pyRoot, "models", "diamondring.glb"), join(publicModels, "diamondring.glb")],
];

for (const [src, dest] of modelAssets) {
  if (!existsSync(src)) {
    console.error(`missing source asset: ${src}`);
    process.exitCode = 1;
    continue;
  }
  cpSync(src, dest);
  console.log(`copied ${src} -> ${dest}`);
}

const mediapipeWasmSrc = join(webRoot, "node_modules", "@mediapipe", "tasks-vision", "wasm");
if (existsSync(mediapipeWasmSrc)) {
  cpSync(mediapipeWasmSrc, publicMediapipeWasm, { recursive: true });
  console.log(`copied ${mediapipeWasmSrc} -> ${publicMediapipeWasm}`);
} else {
  console.warn(`@mediapipe/tasks-vision wasm folder not found yet (run npm install first): ${mediapipeWasmSrc}`);
}

// onnxruntime-web's wasm/mjs runtime files are NOT copied here - they're
// imported directly via Vite's `?url` suffix in geometryNetwork.js instead
// (see that file's comments for why: pointing wasmPaths at a plain public/
// URL string breaks Vite's dev server, since onnxruntime-web dynamically
// import()s its own .mjs glue file, and Vite intercepts any import()-like
// request to a public/ file and tries to transform it).
