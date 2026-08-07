// Port of ring.py's pure-math functions (RingPose Refinement + gating +
// camera-space frame). Every tuned constant carried over as-is from the
// many rounds of live tuning on the Python app - see ring.py's comments
// for the reasoning behind each value.
import { chordLengthMm, estimateRadiusMm } from "./ringGeometry.js";

export const RING_POSITION_FRAC = 0.25;
export const RING_GAP_MM = 0.7;
export const RING_SIZE_SCALE = 0.7;
export const RING_ROLL_OFFSET_DEG = 0.0;
export const RING_TILT_OFFSET_DEG = 0.0;

export const WIDTH_SEARCH_MIN_FRAC = 0.4;
export const WIDTH_SEARCH_MAX_FRAC = 2.2;
export const WIDTH_MIN_EDGE_STRENGTH = 8.0;

export const RING_MIN_STRAIGHTNESS = 0.9;
export const RING_MIN_NEIGHBOR_GAP_FRAC = 0.18;

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1]];
}
function norm2(v) {
  return Math.hypot(v[0], v[1]);
}

/** chord / path_length: 1.0 for a perfectly straight finger, dropping
 * sharply as it bends. pts_px: array of [x,y] (MCP..TIP). */
export function fingerStraightness(ptsPx) {
  let pathLen = 0;
  for (let i = 1; i < ptsPx.length; i++) pathLen += norm2(sub(ptsPx[i], ptsPx[i - 1]));
  if (pathLen < 1e-6) return 0.0;
  const chord = norm2(sub(ptsPx[ptsPx.length - 1], ptsPx[0]));
  return chord / pathLen;
}

/** True if the ring finger has real background gaps to both neighbors at
 * the PIP joints (landmarks 10, 14, 18 - middle/ring/pinky PIP), normalized
 * by the ring finger's own MCP->TIP length (landmarks 13, 16). */
export function fingersSpreadEnough(landmarksPx, minGapFrac = RING_MIN_NEIGHBOR_GAP_FRAC) {
  const ringMcp = landmarksPx[13];
  const ringTip = landmarksPx[16];
  const fingerLen = norm2(sub(ringTip, ringMcp));
  if (fingerLen < 1e-6) return false;

  const middlePip = landmarksPx[10];
  const ringPip = landmarksPx[14];
  const pinkyPip = landmarksPx[18];
  const gapToMiddle = norm2(sub(ringPip, middlePip)) / fingerLen;
  const gapToPinky = norm2(sub(ringPip, pinkyPip)) / fingerLen;
  return gapToMiddle >= minGapFrac && gapToPinky >= minGapFrac;
}

/** ptsPx: polyline in pixel space (MCP..TIP). Returns { center, tangent }
 * (unit) at fraction `frac` of the path's total pixel length, or nulls for
 * a degenerate (zero-length) path. */
export function pointAndTangentAlongPath(ptsPx, frac) {
  const segVecs = [];
  const segLens = [];
  for (let i = 1; i < ptsPx.length; i++) {
    const v = sub(ptsPx[i], ptsPx[i - 1]);
    segVecs.push(v);
    segLens.push(norm2(v));
  }
  const total = segLens.reduce((a, b) => a + b, 0);
  if (total < 1e-6) return { center: null, tangent: null };

  const cum = [0];
  for (const l of segLens) cum.push(cum[cum.length - 1] + l);
  const target = Math.min(Math.max(frac, 0.0), 1.0) * total;

  let segIdx = 0;
  while (segIdx < segLens.length - 1 && cum[segIdx + 1] < target) segIdx++;
  const segLen = Math.max(segLens[segIdx], 1e-6);
  const tLocal = (target - cum[segIdx]) / segLen;

  const center = [ptsPx[segIdx][0] + tLocal * segVecs[segIdx][0], ptsPx[segIdx][1] + tLocal * segVecs[segIdx][1]];
  const tangent = [segVecs[segIdx][0] / segLen, segVecs[segIdx][1] / segLen];
  return { center, tangent };
}

/** RingPose Refinement stage. networkOutput: { centerlineMm, fingerLengthMm }
 * from geometryNetwork.js. Returns null if the pose can't be determined
 * this frame, else { centerPx, tangentPx, radiusMm, scalePxPerMm }. */
export function refineRingPose(networkOutput, detection, positionFrac = RING_POSITION_FRAC) {
  const ptsPx = detection.ringFingerPx; // (4,2) MCP,PIP,DIP,TIP
  const { center: centerPx, tangent: tangentPx } = pointAndTangentAlongPath(ptsPx, positionFrac);
  if (centerPx === null) return null;

  const chordMm = chordLengthMm(networkOutput.centerlineMm);
  const chordPx = norm2(sub(ptsPx[ptsPx.length - 1], ptsPx[0]));
  if (chordMm < 1e-6 || chordPx < 1e-6) return null;
  const scalePxPerMm = chordPx / chordMm;

  const radiusMm = estimateRadiusMm(networkOutput.fingerLengthMm, positionFrac);

  return { centerPx, tangentPx, radiusMm, scalePxPerMm };
}

/** grayData: Uint8ClampedArray/Float32Array of grayscale values, one per
 * pixel, row-major (width x height) - precompute once per frame from the
 * source canvas's ImageData, not per-call. Scans perpendicular to the
 * finger from centerPx, looking for the strongest grayscale-gradient edge
 * within a window around priorHalfwidthPx. Returns the average of the two
 * sides found, or priorHalfwidthPx unchanged if no edge strong enough to
 * trust turns up on either side. */
export function measureFingerHalfwidthPx(
  grayData, width, height, centerPx, perpDir, priorHalfwidthPx,
  minFrac = WIDTH_SEARCH_MIN_FRAC, maxFrac = WIDTH_SEARCH_MAX_FRAC, minEdgeStrength = WIDTH_MIN_EDGE_STRENGTH,
) {
  const lo = Math.max(2, Math.round(priorHalfwidthPx * minFrac));
  const hi = Math.max(lo + 2, Math.round(priorHalfwidthPx * maxFrac));

  function scan(sign) {
    let bestD = null;
    let bestGrad = 0.0;
    let prevVal = null;
    for (let d = 0; d < hi; d++) {
      const px = centerPx[0] + sign * perpDir[0] * d;
      const py = centerPx[1] + sign * perpDir[1] * d;
      const x = Math.round(px);
      const y = Math.round(py);
      if (x < 0 || x >= width || y < 0 || y >= height) break;
      const val = grayData[y * width + x];
      if (prevVal !== null && d >= lo) {
        const grad = Math.abs(val - prevVal);
        if (grad > bestGrad) {
          bestGrad = grad;
          bestD = d;
        }
      }
      prevVal = val;
    }
    return bestD !== null && bestGrad >= minEdgeStrength ? bestD : null;
  }

  const candidates = [scan(1.0), scan(-1.0)].filter((d) => d !== null);
  if (candidates.length === 0) return priorHalfwidthPx;
  return candidates.reduce((a, b) => a + b, 0) / candidates.length;
}

/** A ring's flower/gem is fixed to one physical side of the finger - it
 * doesn't spin to face the camera as the hand turns over. Uses the 2D
 * winding of wrist->index_mcp->pinky_mcp (landmarks 0, 5, 17): its sign
 * flips when the hand flips over. That winding is also anatomically
 * mirrored between a left and a right hand, so `handedness` corrects for
 * that - see ring.py's docstring for the full reasoning (confirmed via a
 * mirrored-landmark test that this correction is necessary and correct). */
export function palmFacingSign(landmarksPx, handedness) {
  const wrist = landmarksPx[0];
  const indexMcp = landmarksPx[5];
  const pinkyMcp = landmarksPx[17];
  const v1 = sub(indexMcp, wrist);
  const v2 = sub(pinkyMcp, wrist);
  const crossZ = v1[0] * v2[1] - v1[1] * v2[0];
  const handSign = handedness === "Right" ? 1.0 : -1.0;
  return crossZ * handSign >= 0 ? 1.0 : -1.0;
}

function normalize3(v) {
  const len = Math.max(Math.hypot(v[0], v[1], v[2]), 1e-9);
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** Builds an orthonormal (tangent, normal, binormal) frame where
 * tangent/normal are the real on-screen finger direction and its in-plane
 * perpendicular, and binormal is a synthetic depth axis this module owns
 * and defines consistently - see ring.py's docstring for the full
 * reasoning (this fixed the "flat/2D" and "squashed loop" bugs from
 * trusting the ONNX model's own ambiguous 3D axes). Returns
 * { tangent3, normal3, binormal3 }, each [x,y,z]. */
export function cameraSpaceFrame(tangentPx, flip = 1.0, rollDeg = RING_ROLL_OFFSET_DEG, tiltDeg = RING_TILT_OFFSET_DEG) {
  const u = (() => {
    const len = Math.max(norm2(tangentPx), 1e-9);
    return [tangentPx[0] / len, tangentPx[1] / len];
  })();
  const baseTangent3 = [u[0], u[1], 0.0];
  const baseNormal3 = [flip * -u[1], flip * u[0], 0.0];
  const baseBinormal3 = cross3(baseTangent3, baseNormal3);

  const radR = (rollDeg * Math.PI) / 180;
  const cosR = Math.cos(radR);
  const sinR = Math.sin(radR);
  const normal3 = addScaled(scale3(baseNormal3, cosR), baseBinormal3, sinR);
  let binormal3 = addScaled(scale3(baseNormal3, -sinR), baseBinormal3, cosR);

  const radT = (tiltDeg * Math.PI) / 180;
  const cosT = Math.cos(radT);
  const sinT = Math.sin(radT);
  const tangent3 = addScaled(scale3(baseTangent3, cosT), binormal3, sinT);
  binormal3 = addScaled(scale3(baseTangent3, -sinT), binormal3, cosT);

  return { tangent3: normalize3(tangent3), normal3: normalize3(normal3), binormal3: normalize3(binormal3) };
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function scale3(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}
function addScaled(a, b, s) {
  return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
}
