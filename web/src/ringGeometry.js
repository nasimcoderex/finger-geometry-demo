// Simplified port of geometry_engine.py. The full Python version fits a
// cubic spline through the ONNX model's 50 centerline points and resamples
// at equal arc-length spacing (for the RMF frame + finger-tube wireframe
// panel) - but main.py dropped that wireframe panel, and ring.py only ever
// reads two things from the result: the chord length (centerline[0] to
// centerline[-1]) and the radius at one arc-length fraction (a formula
// depending only on total length, not the spline). Both are computed
// directly here with no spline library needed - see the plan's Context
// section for why this is a safe simplification, not a missing feature.

const RADIUS_BASE_TO_LENGTH_RATIO = 0.064; // radius at MCP (arc-length fraction t=0)
const RADIUS_TIP_TO_LENGTH_RATIO = 0.042;  // radius at TIP (arc-length fraction t=1)
const MIN_RADIUS_MM = 3.0;
const MAX_RADIUS_MM = 14.0;

function clamp(x, lo, hi) {
  return Math.min(Math.max(x, lo), hi);
}

function smoothstep(t) {
  return 3 * t * t - 2 * t * t * t;
}

/** totalLengthMm: the ONNX model's own finger_length output, used here in
 * place of the Python version's spline arc-length total (see module
 * docstring). t: fraction along the finger, 0=MCP..1=TIP. */
export function estimateRadiusMm(totalLengthMm, t) {
  const rBase = clamp(RADIUS_BASE_TO_LENGTH_RATIO * totalLengthMm, MIN_RADIUS_MM, MAX_RADIUS_MM);
  const rTip = clamp(RADIUS_TIP_TO_LENGTH_RATIO * totalLengthMm, MIN_RADIUS_MM, MAX_RADIUS_MM);
  return rBase + (rTip - rBase) * smoothstep(t);
}

/** centerlineMm: flat Float32Array(150), row-major (50,3) mm, MCP-relative -
 * the ONNX model's raw "centerline" output. Returns the straight-line
 * distance between its first and last points (index 0 and 49). */
export function chordLengthMm(centerlineMm) {
  const dx = centerlineMm[49 * 3] - centerlineMm[0];
  const dy = centerlineMm[49 * 3 + 1] - centerlineMm[1];
  const dz = centerlineMm[49 * 3 + 2] - centerlineMm[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
