// Port of predictor.py: takes the full mirrored frame + 4 ring-finger pixel
// landmarks, reproduces the exact crop/preprocessing the ONNX model was
// trained with (margin 2.0x bbox, no augmentation jitter), and runs it.
//
// Model I/O (unchanged from the Python side):
//   input  "image"          (1,3,256,256) float32, ImageNet-normalized
//   output "centerline"     (1,50,3) mm, MCP-relative
//   output "orientation"    (1,4) unit quaternion (w,x,y,z)
//   output "finger_length"  (1,) mm
// "/wasm" subpath: the WASM-only build (excludes webgpu/webgl/jsep provider
// code this app never uses) - the default `onnxruntime-web` entry pulls in
// every backend, and Vite's bundler statically discovers and bundles their
// wasm binaries too (confirmed: a stray 27MB jsep.wasm showed up in a build
// before this fix), even though wasmPaths below never points at them.
import * as ort from "onnxruntime-web/wasm";
// onnxruntime-web dynamically import()s its own .mjs glue file at runtime.
// Pointing wasmPaths at a plain /public URL string breaks Vite's dev server
// (it intercepts any import()-like request and tries to run it through its
// transform pipeline, which public/ files aren't meant to go through -
// confirmed by hitting exactly this error). The `?url` suffix imports the
// file as a pre-resolved asset URL instead, which Vite explicitly exempts
// from that pipeline - works identically in dev and production builds.
// Both files must come from the exact same onnxruntime-web version.
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortWasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";

ort.env.wasm.wasmPaths = {
  "ort-wasm-simd-threaded.wasm": ortWasmUrl,
  "ort-wasm-simd-threaded.mjs": ortWasmMjsUrl,
};

export const INPUT_IMAGE_SIZE = 256;
export const CROP_MARGIN_FACTOR = 2.0; // matches predictor.py's CROP_MARGIN_FACTOR
export const MIN_CROP_SIZE_PX = 20;

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export function computeCropBox(ringFingerPx, marginFactor = CROP_MARGIN_FACTOR) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of ringFingerPx) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  const cx = (x0 + x1) / 2.0;
  const cy = (y0 + y1) / 2.0;
  const boxSize = Math.max(x1 - x0, y1 - y0, MIN_CROP_SIZE_PX);
  const half = (boxSize * marginFactor) / 2.0;
  return [cx - half, cy - half, cx + half, cy + half]; // x0,y0,x1,y1
}

export class GeometryPredictor {
  constructor() {
    this._session = null;
    this._cropCanvas = document.createElement("canvas");
    this._cropCanvas.width = INPUT_IMAGE_SIZE;
    this._cropCanvas.height = INPUT_IMAGE_SIZE;
    this._cropCtx = this._cropCanvas.getContext("2d", { willReadFrequently: true });
  }

  async init() {
    this._session = await ort.InferenceSession.create("/models/model.onnx", {
      executionProviders: ["wasm"],
    });
  }

  /** source: the mirrored frame canvas (same one hand detection ran on).
   * ringFingerPx: (4,2) [MCP,PIP,DIP,TIP] pixel coords in that source.
   * Returns { centerlineMm: Float32Array(150), orientationQuat: Float32Array(4),
   *   fingerLengthMm: number, cropBox: [x0,y0,x1,y1] }. */
  async predict(source, ringFingerPx) {
    const box = computeCropBox(ringFingerPx);
    const [x0, y0, x1, y1] = box;
    const boxW = x1 - x0;
    const boxH = y1 - y0;

    // PIL's Image.crop() pads out-of-bounds regions with black rather than
    // erroring/clipping the crop size - replicate that by clearing to black
    // first, then drawImage naturally only paints the valid overlapping
    // region (browsers clip the source rect to the image's real bounds).
    this._cropCtx.fillStyle = "black";
    this._cropCtx.fillRect(0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
    this._cropCtx.drawImage(source, x0, y0, boxW, boxH, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);

    const { data } = this._cropCtx.getImageData(0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
    const chw = new Float32Array(3 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE);
    const plane = INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE;
    for (let i = 0; i < plane; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;
      chw[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      chw[plane + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      chw[2 * plane + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    const imageTensor = new ort.Tensor("float32", chw, [1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE]);
    const outputs = await this._session.run({ image: imageTensor });

    return {
      centerlineMm: outputs.centerline.data, // (50*3,) flat, row-major (50,3)
      orientationQuat: outputs.orientation.data, // (4,) w,x,y,z
      fingerLengthMm: outputs.finger_length.data[0],
      cropBox: box,
    };
  }
}
