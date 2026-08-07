// Direct port of ring.py's _OneEuroFilter / TemporalFilter. Same tuned
// constants (min_cutoff=0.6, beta=0.25) found through many rounds of live
// tuning on the Python app - see ring.py's comments for the jitter/lag
// tradeoff measurements that led to these values.
export const ONE_EURO_MIN_CUTOFF = 0.6;
export const ONE_EURO_BETA = 0.25;

class OneEuroFilterChannel {
  constructor(minCutoff = ONE_EURO_MIN_CUTOFF, beta = ONE_EURO_BETA, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this._xPrev = null;
    this._dxPrev = 0.0;
    this._tPrev = null;
  }

  static _alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, t) {
    if (this._tPrev === null) {
      this._xPrev = x;
      this._tPrev = t;
      this._dxPrev = 0.0;
      return x;
    }
    const dt = Math.max(t - this._tPrev, 1e-6);

    const dx = (x - this._xPrev) / dt;
    const aD = OneEuroFilterChannel._alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1.0 - aD) * this._dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuroFilterChannel._alpha(cutoff, dt);
    const xHat = a * x + (1.0 - a) * this._xPrev;

    this._xPrev = xHat;
    this._dxPrev = dxHat;
    this._tPrev = t;
    return xHat;
  }
}

/** One Euro Filter over the refined ring pose (6 independent channels:
 * center x/y, tangent x/y, radius_mm, scale_px_per_mm) plus one more for
 * the pixel-measured half-width. One instance per session. */
export class TemporalFilter {
  constructor(minCutoff = ONE_EURO_MIN_CUTOFF, beta = ONE_EURO_BETA) {
    this._minCutoff = minCutoff;
    this._beta = beta;
    this._filters = null;
    this._halfwidthFilter = null;
  }

  update(centerPx, tangentPx, radiusMm, scalePxPerMm) {
    if (this._filters === null) {
      this._filters = Array.from({ length: 6 }, () => new OneEuroFilterChannel(this._minCutoff, this._beta));
    }
    const t = performance.now() / 1000;
    const raw = [centerPx[0], centerPx[1], tangentPx[0], tangentPx[1], radiusMm, scalePxPerMm];
    const out = raw.map((v, i) => this._filters[i].filter(v, t));

    const center = [out[0], out[1]];
    let tangent = [out[2], out[3]];
    const tangentLen = Math.max(Math.hypot(tangent[0], tangent[1]), 1e-9);
    tangent = [tangent[0] / tangentLen, tangent[1] / tangentLen];
    return { center, tangent, radiusMm: out[4], scalePxPerMm: out[5] };
  }

  /** Separate filter for the pixel-measured half-width - that measurement
   * has its own frame-to-frame noise independent of the pose smoothing above. */
  smoothHalfwidthPx(value) {
    if (this._halfwidthFilter === null) {
      this._halfwidthFilter = new OneEuroFilterChannel(this._minCutoff, this._beta);
    }
    return this._halfwidthFilter.filter(value, performance.now() / 1000);
  }

  reset() {
    this._filters = null;
    this._halfwidthFilter = null;
  }
}
