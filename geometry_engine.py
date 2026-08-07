"""Standalone reimplementation of the "geometry engine" step: turns the
ONNX model's 3 raw outputs (centerline_mm, orientation_quat, finger_length_mm)
into the full per-point profile a mesh needs (tangent/normal/binormal frames
+ radius at all 50 points).

This is the same math as F:\\finger-geometry-model's
training/targets.py:reconstruct_full_geometry + label_generator/orientation.py
+ label_generator/radius.py - copied rather than imported so this project has
no dependency on that repo's layout or its torch/torchvision install (none of
this math needs torch; it's pure numpy/scipy). Any behavior change there
should be mirrored here by hand.

Units: the ONNX model's own outputs are millimeters (centerline_mm,
finger_length_mm). Everything in this module stays in millimeters too, unlike
the training repo (which works in meters) - there's no reason to convert
back and forth in a demo that only ever displays mm.
"""
import numpy as np
from scipy.interpolate import splev, splprep

CENTERLINE_N_SAMPLES = 50

# label_generator/radius.py's fixed taper formula, unchanged - see that
# module's docstring for why radius depends only on total arc length, not
# per-sample image evidence (a known, documented limitation, not a bug here).
RADIUS_BASE_TO_LENGTH_RATIO = 0.064   # radius at MCP (arc-length fraction t=0)
RADIUS_TIP_TO_LENGTH_RATIO = 0.042    # radius at TIP (arc-length fraction t=1)
MIN_RADIUS_MM = 3.0
MAX_RADIUS_MM = 14.0


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


def _smoothstep(t):
    return 3 * t**2 - 2 * t**3


def estimate_radius_mm(arc_length_mm):
    total = arc_length_mm[-1]
    r_base = np.clip(RADIUS_BASE_TO_LENGTH_RATIO * total, MIN_RADIUS_MM, MAX_RADIUS_MM)
    r_tip = np.clip(RADIUS_TIP_TO_LENGTH_RATIO * total, MIN_RADIUS_MM, MAX_RADIUS_MM)
    t = arc_length_mm / total
    return r_base + (r_tip - r_base) * _smoothstep(t)


def _initial_normal(tangent0):
    for ref in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        if abs(np.dot(ref, tangent0)) < 0.9:
            n = ref - np.dot(ref, tangent0) * tangent0
            return n / np.linalg.norm(n)
    raise RuntimeError("no reference vector found that isn't parallel to the initial tangent")


def propagate_rmf(centerline, tangents, seed_normal):
    """Double Reflection Method (Wang/Juttler/Zheng/Liu 2008) - curvature-
    independent frame propagation, so it doesn't flip on the near-straight
    spans a finger mostly consists of. See label_generator/orientation.py in
    the training repo for the full derivation notes."""
    n = len(centerline)
    normals = np.zeros((n, 3))
    n0 = seed_normal - np.dot(seed_normal, tangents[0]) * tangents[0]
    normals[0] = n0 / np.linalg.norm(n0)

    for i in range(n - 1):
        v1 = centerline[i + 1] - centerline[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            normals[i + 1] = normals[i]
            continue
        r_l = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        t_l = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1

        v2 = tangents[i + 1] - t_l
        c2 = np.dot(v2, v2)
        n_next = r_l if c2 < 1e-20 else r_l - (2.0 / c2) * np.dot(v2, r_l) * v2

        n_next = n_next - np.dot(n_next, tangents[i + 1]) * tangents[i + 1]
        normals[i + 1] = n_next / np.linalg.norm(n_next)

    return normals


def reconstruct_full_geometry(centerline_mm, orientation_quat, finger_length_mm):
    """centerline_mm: (50,3) network output. orientation_quat: (4,) unit
    quaternion (w,x,y,z) for the base frame at MCP. finger_length_mm: scalar,
    the network's independent length estimate (kept alongside, not forced to
    match centerline_mm's own arc length - see training repo's
    targets.py:reconstruct_full_geometry docstring for why).

    Returns a dict: centerline, arc_length, tangent, normal, binormal, radius
    (all mm, all (50,3) or (50,)), plus finger_length_mm passed through.
    """
    tck, _ = splprep(centerline_mm.T, k=3, s=0)
    u_dense = np.linspace(0.0, 1.0, 2000)
    pts_dense = np.array(splev(u_dense, tck)).T
    seg_lengths = np.linalg.norm(np.diff(pts_dense, axis=0), axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    target_lengths = np.linspace(0.0, total_length, CENTERLINE_N_SAMPLES)
    u_samples = np.interp(target_lengths, cum_length, u_dense)
    centerline = np.array(splev(u_samples, tck)).T
    arc_length = target_lengths

    tangents_raw = np.array(splev(u_samples, tck, der=1)).T
    tangent = tangents_raw / np.linalg.norm(tangents_raw, axis=1, keepdims=True)

    seed_normal = quaternion_to_rotation_matrix(orientation_quat)[:, 1]
    normal = propagate_rmf(centerline, tangent, seed_normal)
    binormal = np.cross(tangent, normal)

    radius = estimate_radius_mm(arc_length)

    return {
        "centerline": centerline,
        "arc_length": arc_length,
        "tangent": tangent,
        "normal": normal,
        "binormal": binormal,
        "radius": radius,
        "finger_length_mm": float(finger_length_mm),
    }
