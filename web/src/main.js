// camera -> MediaPipe hand detection -> finger-geometry ONNX model ->
// ring-pose math -> Three.js ring overlay. Browser port of main.py.
import { HandDetector } from "./handTracking.js";
import { GeometryPredictor } from "./geometryNetwork.js";
import {
  fingerStraightness, fingersSpreadEnough, refineRingPose, measureFingerHalfwidthPx,
  palmFacingSign, cameraSpaceFrame,
  RING_POSITION_FRAC, RING_GAP_MM, RING_SIZE_SCALE, RING_ROLL_OFFSET_DEG, RING_TILT_OFFSET_DEG,
  RING_MIN_STRAIGHTNESS, RING_MIN_NEIGHBOR_GAP_FRAC,
} from "./ringPose.js";
import { TemporalFilter } from "./oneEuroFilter.js";
import { RingRenderer } from "./ringRenderer.js";
import { updateBanner } from "./ui.js";

const video = document.getElementById("video");
const frameCanvas = document.getElementById("frameCanvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });
const ringCanvas = document.getElementById("ringCanvas");
const overlay = document.getElementById("overlay"); // temporary landmark debug - see drawLandmarkDebug
const overlayCtx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const bannerEl = document.getElementById("banner");

const handDetector = new HandDetector();
const geometryPredictor = new GeometryPredictor();
const ringRenderer = new RingRenderer(ringCanvas);
const ringFilter = new TemporalFilter();

async function openCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

function drawMirroredFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return false;
  for (const c of [frameCanvas, ringCanvas, overlay]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  // mirror the actual pixel data (not just the CSS display) - single source
  // of truth for every downstream stage, matching main.py's
  // cv2.flip(frame_bgr, 1) being the first thing that happens to the frame.
  // Every sign convention tuned in the Python app assumes this same
  // mirrored coordinate space.
  frameCtx.save();
  frameCtx.translate(w, 0);
  frameCtx.scale(-1, 1);
  frameCtx.drawImage(video, 0, 0, w, h);
  frameCtx.restore();
  return true;
}

// cv2.cvtColor(..., COLOR_BGR2GRAY) equivalent (ITU-R BT.601 luma weights -
// same formula regardless of channel storage order).
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  return gray;
}

// TEMPORARY debug overlay (dots on all 21 landmarks, highlighted ring
// finger) - kept for now to help confirm tracking while the ring render is
// being verified live; remove once confirmed working, matching how the
// Python app's debug overlay was cleaned up once the ring itself worked.
function drawLandmarkDebug(detection) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!detection) return;
  overlayCtx.fillStyle = "rgba(150,150,150,0.7)";
  for (const [x, y] of detection.landmarksPx) {
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 2, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

async function processFrame() {
  const w = frameCanvas.width;
  const h = frameCanvas.height;
  const detection = handDetector.detect(frameCanvas, w, h);
  drawLandmarkDebug(detection);

  if (!detection) {
    ringFilter.reset();
    ringRenderer.renderFrame(null);
    updateBanner(bannerEl, "no_hand");
    return { statusLine: "no hand detected" };
  }

  if (fingerStraightness(detection.ringFingerPx) < RING_MIN_STRAIGHTNESS) {
    ringRenderer.renderFrame(null);
    updateBanner(bannerEl, "not_straight");
    return { statusLine: `hand: ${detection.handedness}` };
  }
  if (!fingersSpreadEnough(detection.landmarksPx, RING_MIN_NEIGHBOR_GAP_FRAC)) {
    ringRenderer.renderFrame(null);
    updateBanner(bannerEl, "fingers_together");
    return { statusLine: `hand: ${detection.handedness}` };
  }

  const networkOutput = await geometryPredictor.predict(frameCanvas, detection.ringFingerPx);
  const pose = refineRingPose(networkOutput, detection, RING_POSITION_FRAC);
  if (pose === null) {
    ringRenderer.renderFrame(null);
    updateBanner(bannerEl, null);
    return { statusLine: `hand: ${detection.handedness}` };
  }

  const filtered = ringFilter.update(pose.centerPx, pose.tangentPx, pose.radiusMm, pose.scalePxPerMm);

  const priorHalfwidthPx = (filtered.radiusMm + RING_GAP_MM) * filtered.scalePxPerMm;
  const perpDir = [-filtered.tangent[1], filtered.tangent[0]];
  const perpLen = Math.max(Math.hypot(perpDir[0], perpDir[1]), 1e-9);
  perpDir[0] /= perpLen;
  perpDir[1] /= perpLen;

  const grayData = toGrayscale(frameCtx.getImageData(0, 0, w, h));
  let measuredHalfwidthPx = measureFingerHalfwidthPx(grayData, w, h, filtered.center, perpDir, priorHalfwidthPx);
  measuredHalfwidthPx = ringFilter.smoothHalfwidthPx(measuredHalfwidthPx);

  const targetInnerRadiusPx = measuredHalfwidthPx * RING_SIZE_SCALE;
  if (targetInnerRadiusPx < 2.0) {
    ringRenderer.renderFrame(null);
    updateBanner(bannerEl, null);
    return { statusLine: `hand: ${detection.handedness}` };
  }

  const palmFlip = palmFacingSign(detection.landmarksPx, detection.handedness);
  const { tangent3, normal3, binormal3 } = cameraSpaceFrame(
    filtered.tangent, palmFlip, RING_ROLL_OFFSET_DEG, RING_TILT_OFFSET_DEG,
  );
  ringRenderer.renderFrame({ centerPx: filtered.center, tangent3, normal3, binormal3, scalePx: targetInnerRadiusPx });
  updateBanner(bannerEl, null);
  return {
    statusLine: `hand: ${detection.handedness}\nfinger_length: ${networkOutput.fingerLengthMm.toFixed(1)} mm\nradius: ${pose.radiusMm.toFixed(1)} mm`,
  };
}

let fpsN = 0;
let fpsT0 = performance.now();
let fps = 0;

async function loop() {
  try {
    const ok = drawMirroredFrame();
    if (ok) {
      ringRenderer.resize(frameCanvas.width, frameCanvas.height);
      const { statusLine } = await processFrame();

      fpsN += 1;
      const now = performance.now();
      if (now - fpsT0 >= 500) {
        fps = (fpsN * 1000) / (now - fpsT0);
        fpsN = 0;
        fpsT0 = now;
      }
      statusEl.textContent = `FPS ${fps.toFixed(1)}\n${statusLine}`;
    }
  } catch (err) {
    console.error(err);
  }
  requestAnimationFrame(loop); // only schedule the next frame once this one's async work is done
}

async function main() {
  statusEl.textContent = "requesting camera...";
  await openCamera();
  statusEl.textContent = "loading hand tracking model...";
  await handDetector.init();
  statusEl.textContent = "loading finger geometry model...";
  await geometryPredictor.init();
  statusEl.textContent = "loading ring model...";
  await ringRenderer.loadRingAsset("/models/diamondring.glb");
  requestAnimationFrame(loop);
}

main().catch((err) => {
  statusEl.textContent = `startup error: ${err.message}`;
  console.error(err);
});
