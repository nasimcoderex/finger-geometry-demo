"""RingPose Refinement + Temporal Filter + renderer stages, standing in for
the Three.js/GLB half of the target pipeline while rendering stays in
Python/OpenCV. Consumes the mesh built by ring_model.py (the real .glb file
on disk).

RingPose Refinement here is analytic, not a second trained network: this
project's only custom model (model.onnx) is the Finger Geometry Network, so
there's no dataset to train a dedicated refinement model against. This stage
plays the same *role* - turning raw detection + geometry-engine output into
a clean ring pose (position, orientation, metric size) - just implemented
in code.

Camera-space frame (the fix for the earlier "flat/2D" and "squashed loop"
attempts): those relied on the ONNX model's own mm-space X/Y/Z axes lining
up with the camera, an assumption that's undocumented anywhere in this repo
and turned out wrong. Here the frame is instead built from what's actually
known to be correct - the real on-screen finger direction (tangent_px, from
MediaPipe landmarks) - plus a synthetic depth axis that this module defines
itself (cross(tangent, in-plane-normal), always exactly (0,0,1) by
construction). That depth axis has no ambiguity because nothing external is
trusted for its sign or meaning; it only has to be self-consistent within
this one function, which it is. Real occlusion (mesh triangles facing away
from camera get culled) and real per-triangle shading fall out of that for
free, instead of being faked with 2D arc tricks.
"""
import time

import numpy as np
import cv2

RING_POSITION_FRAC = 0.22   # fraction along the MCP->PIP->DIP->TIP landmark path
RING_GAP_MM = 0.7           # clearance between finger surface and inner band edge (in mm-equivalent px)
RING_SIZE_SCALE = 0.8       # final polish knob on top of the measured width; 1.0 = trust the measurement
# Rotation around the finger's own axis (tangent3) - controls where around the
# finger's circumference the gem/setting sits. 0 = whatever the asset's own
# local-space "theta=90" convention lands on for a given hand pose. Positive
# degrees rotate the same direction as standard math angle convention (using
# normal3/binormal3 as the local x/y of that rotation plane).
RING_ROLL_OFFSET_DEG = 0.0

# One Euro Filter (Casiez et al. 2012) parameters. A fixed-alpha EMA can't
# win here: smooth enough to kill landmark jitter when the hand is still is
# automatically too slow to track a fast hand move, which reads as the ring
# lagging behind / "floating" free of the finger. One Euro adapts instead -
# min_cutoff sets how much smoothing happens at rest, beta controls how fast
# that smoothing backs off as the tracked value's own speed increases.
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.4

# geometry_engine.py's radius is a fixed taper-ratio heuristic on predicted
# total finger length, and converting it to pixels via the MCP->TIP chord is
# sensitive to how much the finger is foreshortened toward the camera in any
# given frame (a pointed-at-camera pose shrinks the observed chord, silently
# undersizing the ring; a bent finger does the opposite) - a constant
# multiplier can't compensate for an error that changes with pose. Instead,
# the model's radius is used only as a *prior* to center a search window,
# then _measure_finger_halfwidth_px refines it against this frame's actual
# pixels (a gradient edge scan perpendicular to the finger), so sizing
# tracks the real visible finger regardless of pose.
WIDTH_SEARCH_MIN_FRAC = 0.4
WIDTH_SEARCH_MAX_FRAC = 2.2
WIDTH_MIN_EDGE_STRENGTH = 8.0     # grayscale gradient magnitude (0-255 scale) to count as a real edge


class _OneEuroFilter:
    """One scalar channel of a One Euro Filter. See module docstring for why
    this replaced a fixed-alpha EMA."""

    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self._t_prev is None:
            self._x_prev, self._t_prev, self._dx_prev = x, t, 0.0
            return x
        dt = max(t - self._t_prev, 1e-6)

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class TemporalFilter:
    """One Euro Filter over the refined ring pose (6 independent channels:
    center x/y, tangent x/y, radius_mm, scale_px_per_mm) plus one more for
    the pixel-measured half-width. One instance per session, fed a fresh
    sample every frame detection succeeds."""

    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._filters = None
        self._halfwidth_filter = None

    def update(self, center_px, tangent_px, radius_mm, scale_px_per_mm):
        if self._filters is None:
            self._filters = [_OneEuroFilter(self._min_cutoff, self._beta) for _ in range(6)]
        t = time.time()
        raw = (center_px[0], center_px[1], tangent_px[0], tangent_px[1], radius_mm, scale_px_per_mm)
        out = [f.filter(v, t) for f, v in zip(self._filters, raw)]

        center = np.array(out[0:2])
        tangent = np.array(out[2:4])
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        return center, tangent, float(out[4]), float(out[5])

    def smooth_halfwidth_px(self, value):
        """Separate filter for the pixel-measured half-width - that
        measurement has its own frame-to-frame noise (lighting, exact edge
        pixel) independent of the pose smoothing above."""
        if self._halfwidth_filter is None:
            self._halfwidth_filter = _OneEuroFilter(self._min_cutoff, self._beta)
        return self._halfwidth_filter.filter(value, time.time())

    def reset(self):
        self._filters = None
        self._halfwidth_filter = None


def _point_and_tangent_along_path(pts_px, frac):
    """pts_px: (N,2) polyline in pixel space (MCP..TIP). Returns (center_px,
    unit_tangent_px) at fraction `frac` of the path's total pixel length."""
    seg_vecs = np.diff(pts_px, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total = seg_lens.sum()
    if total < 1e-6:
        return None, None

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    target = np.clip(frac, 0.0, 1.0) * total
    seg_idx = min(np.searchsorted(cum, target, side="right") - 1, len(seg_lens) - 1)
    seg_idx = max(seg_idx, 0)
    seg_len = max(seg_lens[seg_idx], 1e-6)
    t_local = (target - cum[seg_idx]) / seg_len

    center = pts_px[seg_idx] + t_local * seg_vecs[seg_idx]
    tangent = seg_vecs[seg_idx] / seg_len
    return center, tangent


def refine_ring_pose(geometry, detection, position_frac=RING_POSITION_FRAC):
    """RingPose Refinement stage. Returns None if the pose can't be
    determined this frame (degenerate detection), else a dict with
    center_px (2,), tangent_px (2, unit), radius_mm (float, the finger's
    local cross-sectional radius), scale_px_per_mm (float)."""
    pts_px = detection["ring_finger_px"]  # (4,2) MCP,PIP,DIP,TIP - real detections
    center_px, tangent_px = _point_and_tangent_along_path(pts_px, position_frac)
    if center_px is None:
        return None

    chord_mm = np.linalg.norm(geometry["centerline"][-1] - geometry["centerline"][0])
    chord_px = np.linalg.norm(pts_px[-1] - pts_px[0])
    if chord_mm < 1e-6 or chord_px < 1e-6:
        return None
    scale_px_per_mm = chord_px / chord_mm

    idx = int(round(position_frac * (len(geometry["radius"]) - 1)))
    radius_mm = float(geometry["radius"][idx])

    return {
        "center_px": center_px,
        "tangent_px": tangent_px,
        "radius_mm": radius_mm,
        "scale_px_per_mm": scale_px_per_mm,
    }


def _measure_finger_halfwidth_px(gray_frame, center_px, perp_dir, prior_halfwidth_px,
                                  min_frac=WIDTH_SEARCH_MIN_FRAC, max_frac=WIDTH_SEARCH_MAX_FRAC,
                                  min_edge_strength=WIDTH_MIN_EDGE_STRENGTH):
    """Scans perpendicular to the finger from center_px, looking for the
    strongest grayscale-gradient edge (the finger/background boundary)
    within a window around the model-based prior. Returns the average of
    the two sides found, or `prior_halfwidth_px` unchanged if no edge strong
    enough to trust turns up on either side (e.g. skin-toned background,
    low contrast) - a bounded refinement, not a blind replacement."""
    h, w = gray_frame.shape[:2]
    lo = max(2, int(round(prior_halfwidth_px * min_frac)))
    hi = max(lo + 2, int(round(prior_halfwidth_px * max_frac)))

    def scan(sign):
        best_d, best_grad, prev_val = None, 0.0, None
        for d in range(0, hi):
            p = center_px + sign * perp_dir * d
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < w and 0 <= y < h):
                break
            val = float(gray_frame[y, x])
            if prev_val is not None and d >= lo:
                grad = abs(val - prev_val)
                if grad > best_grad:
                    best_grad, best_d = grad, d
            prev_val = val
        return float(best_d) if best_d is not None and best_grad >= min_edge_strength else None

    candidates = [d for d in (scan(1.0), scan(-1.0)) if d is not None]
    return float(np.mean(candidates)) if candidates else prior_halfwidth_px


def _palm_facing_sign(landmarks_px):
    """A ring's flower is fixed to one physical side of the finger - it
    doesn't spin to face the camera as the hand turns over. This needs to
    know which way the hand is actually rotated so the flower can flip to
    the far (hidden) side when the palm turns to face the camera instead of
    the back of the hand. Uses the 2D winding of wrist->index_mcp->pinky_mcp
    (landmarks 0, 5, 17): its sign flips when the hand flips over, and it's
    computed purely from real 2D detections - no dependence on the ONNX
    model's ambiguous 3D output. Sign convention is a guess (matches the
    poses tested); flip it here if a hand-flip still doesn't move the flower
    to the far side."""
    wrist, index_mcp, pinky_mcp = landmarks_px[0], landmarks_px[5], landmarks_px[17]
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    return 1.0 if cross_z >= 0 else -1.0


def _camera_space_frame(tangent_px, flip=1.0, roll_deg=RING_ROLL_OFFSET_DEG):
    """Builds an orthonormal (tangent, normal, binormal) frame where
    tangent/normal are the real on-screen finger direction and its in-plane
    perpendicular, and binormal = cross(tangent, normal) is, by
    construction, always exactly (0, 0, 1) - a synthetic depth axis this
    module owns and defines consistently, unlike the model's mm-space Z.
    `flip` (+-1, from _palm_facing_sign) rotates this frame 180 deg around
    the tangent axis when the hand has turned over, so the mesh's fixed
    "front" (local theta=90) tracks the hand's real rotation instead of
    always facing the camera regardless of pose. `roll_deg` is a finer
    rotation around that same tangent axis, for nudging exactly where
    around the finger's circumference the mesh's gem/setting lands."""
    u = tangent_px / max(np.linalg.norm(tangent_px), 1e-9)
    tangent3 = np.array([u[0], u[1], 0.0])
    base_normal3 = flip * np.array([-u[1], u[0], 0.0])
    base_binormal3 = np.cross(tangent3, base_normal3)

    rad = np.radians(roll_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    normal3 = cos_r * base_normal3 + sin_r * base_binormal3
    binormal3 = -sin_r * base_normal3 + cos_r * base_binormal3
    return tangent3, normal3, binormal3


def render_ring_overlay(frame_bgr, geometry, detection, ring_mesh,
                         temporal_filter=None, position_frac=RING_POSITION_FRAC,
                         gap_mm=RING_GAP_MM, size_scale=RING_SIZE_SCALE,
                         roll_deg=RING_ROLL_OFFSET_DEG, measurement_frame_bgr=None):
    """Composites the loaded GLB ring mesh onto frame_bgr in-place, wrapped
    around the finger at `position_frac`. ring_mesh: (vertices, faces,
    normals, colors) as returned by ring_model.ensure_ring_glb() - vertices
    in canonical local space (hole axis = local Z, inner radius = 1.0 unit),
    colors (N,4) RGBA 0..1 per vertex (gold band, rose petals, gold-white
    flower center - see ring_model.py). measurement_frame_bgr: the frame to
    scan for the finger's real edge, ideally *before* any overlay drawing
    (landmark dots, boxes) has been painted onto it - those would otherwise
    contaminate the gradient scan. Defaults to frame_bgr if not given."""
    pose = refine_ring_pose(geometry, detection, position_frac)
    if pose is None:
        return

    if temporal_filter is not None:
        center_px, tangent_px, radius_mm, scale_px_per_mm = temporal_filter.update(
            pose["center_px"], pose["tangent_px"], pose["radius_mm"], pose["scale_px_per_mm"])
    else:
        center_px, tangent_px = pose["center_px"], pose["tangent_px"]
        radius_mm, scale_px_per_mm = pose["radius_mm"], pose["scale_px_per_mm"]

    prior_halfwidth_px = (radius_mm + gap_mm) * scale_px_per_mm
    perp_dir = np.array([-tangent_px[1], tangent_px[0]])
    perp_dir = perp_dir / max(np.linalg.norm(perp_dir), 1e-9)

    scan_frame = measurement_frame_bgr if measurement_frame_bgr is not None else frame_bgr
    gray = cv2.cvtColor(scan_frame, cv2.COLOR_BGR2GRAY)
    measured_halfwidth_px = _measure_finger_halfwidth_px(gray, center_px, perp_dir, prior_halfwidth_px)
    if temporal_filter is not None:
        measured_halfwidth_px = temporal_filter.smooth_halfwidth_px(measured_halfwidth_px)

    target_inner_radius_px = measured_halfwidth_px * size_scale
    if target_inner_radius_px < 2.0:
        return

    palm_flip = _palm_facing_sign(detection["landmarks_px"])
    tangent3, normal3, binormal3 = _camera_space_frame(tangent_px, flip=palm_flip, roll_deg=roll_deg)
    # local X -> normal3 (across finger, visible), local Y -> binormal3 (depth,
    # invisible in the 2D projection but used below for shading/occlusion),
    # local Z -> tangent3 (along finger, visible, small spread from band width)
    R = np.stack([normal3, binormal3, tangent3], axis=1)  # local-axis basis, columns

    verts_local, faces, normals_local, colors_local = ring_mesh
    world = (verts_local * target_inner_radius_px) @ R.T  # (N,3): x,y are pixel offsets, z is synthetic depth
    world_px = world[:, :2] + center_px[None, :]
    world_normals = normals_local @ R.T
    colors_bgr = colors_local[:, [2, 1, 0]]  # RGBA -> BGR, alpha dropped (opaque)

    tri_px = world_px[faces]          # (M,3,2)
    tri_depth = world[faces, 2]       # (M,3)
    tri_normal = world_normals[faces].mean(axis=1)  # (M,3), flat-shade approximation
    tri_color = colors_bgr[faces].mean(axis=1)      # (M,3), average of the 3 vertex colors

    view_dir = np.array([0.0, 0.0, 1.0])  # this module's own convention: +Z = toward camera
    facing = tri_normal @ view_dir  # still used below for shading (lighting), not for culling

    # Visibility must be decided by POSITION (which half of the ring's own
    # geometry, near vs far side of the finger), not by surface normal.
    # Normal-based backface culling is right for a thin hollow loop (the
    # original procedural torus) where a triangle's outward-normal direction
    # happens to coincide with which side of the loop it's on - but this
    # real asset has genuine volume/thickness, so its normals point every
    # which way regardless of position. Culling by normal there just shows
    # the outer surface of a solid 3D chunk, which still traces out a
    # complete ring silhouette (confirmed by rendering it standalone - see
    # conversation) instead of hiding the far side behind the finger.
    all_avg_depth = tri_depth.mean(axis=1)
    visible = all_avg_depth > 0.0
    if not visible.any():
        return

    avg_depth = all_avg_depth[visible]
    order = np.argsort(avg_depth)  # farthest (most negative) first, painter's algorithm

    vis_tri_px = tri_px[visible][order]
    vis_facing = np.clip(facing[visible][order], 0.0, 1.0)
    vis_diffuse = 0.25 + 0.75 * vis_facing
    vis_color = tri_color[visible][order]

    # camera-collocated light (light_dir == view_dir) means the Blinn half-vector
    # equals view_dir too, so N.H reduces to N.V (= facing) - a high power of that
    # gives a tight specular streak instead of the broad diffuse falloff, which
    # is what makes flat metal look "flat" versus "polished".
    vis_spec = vis_facing ** 40

    white = np.array([255.0, 255.0, 255.0])
    for tri, diffuse, spec, base in zip(vis_tri_px, vis_diffuse, vis_spec, vis_color):
        color = tuple(int(c) for c in np.clip(base * 255.0 * diffuse + white * spec * 0.65, 0, 255))
        cv2.fillConvexPoly(frame_bgr, tri.astype(np.int32), color, cv2.LINE_AA)
