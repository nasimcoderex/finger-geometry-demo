// Port of hand_detector.py: wraps MediaPipe's HandLandmarker (Tasks API,
// self-hosted WASM + .task model - see scripts/copy-assets.mjs) and
// extracts the 4 ring-finger landmarks the ONNX model's crop box needs.
//
// Landmark topology (same as Python, same underlying MediaPipe model): 21
// points, 0=wrist, 5/9/13/17=index/middle/ring/pinky MCP, each finger
// MCP->PIP->DIP->TIP in order. Ring finger = indices [13,14,15,16].
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export const RING_FINGER_INDICES = [13, 14, 15, 16]; // MCP, PIP, DIP, TIP

export class HandDetector {
  constructor() {
    this._landmarker = null;
    this._timestampMs = 0;
  }

  async init({ maxHands = 1, minDetectionConfidence = 0.6 } = {}) {
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    this._landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/models/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: maxHands,
      minHandDetectionConfidence: minDetectionConfidence,
    });
  }

  /** source: a canvas/video/ImageBitmap already in the mirrored coordinate
   * space (see main.js) - w/h are that source's pixel dimensions, used to
   * convert MediaPipe's normalized [0,1] landmark coords to pixels, same as
   * Python's `pts = lm.x*w, lm.y*h`. Returns null if no hand found, else
   * { landmarksPx: Float32Array-like (21,2), ringFingerPx: (4,2),
   *   handedness: "Left"|"Right" }. */
  detect(source, w, h) {
    // timestamps must strictly increase - mirrors Python's `_timestamp_ms += 1`
    this._timestampMs += 1;
    const result = this._landmarker.detectForVideo(source, this._timestampMs);
    if (!result.landmarks || result.landmarks.length === 0) return null;

    const landmarks = result.landmarks[0]; // first detected hand only
    const landmarksPx = landmarks.map((lm) => [lm.x * w, lm.y * h]);
    const ringFingerPx = RING_FINGER_INDICES.map((i) => landmarksPx[i]);
    const handedness = result.handedness?.[0]?.[0]?.categoryName ?? "?";

    return { landmarksPx, ringFingerPx, handedness };
  }

  close() {
    this._landmarker?.close();
  }
}
