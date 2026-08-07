"""Cheap wireframe renderer for the reconstructed finger geometry - a fixed
isometric-style projection drawn with cv2 polylines, so the live demo needs
no extra 3D library (no Open3D/matplotlib window juggling) and stays at
camera frame rate.
"""
import numpy as np
import cv2

from mesh import DEFAULT_TIP_CAP_RINGS, build_finger_mesh

_ROT_X_DEG = 20.0
_ROT_Y_DEG = 35.0


def _rotation():
    tx, ty = np.radians(_ROT_X_DEG), np.radians(_ROT_Y_DEG)
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    return Ry @ Rx


_R = _rotation()


def render_wireframe(geometry, panel_size=420, sides=20, margin_frac=0.12):
    """geometry: dict from geometry_engine.reconstruct_full_geometry.
    Returns a panel_size x panel_size x 3 BGR uint8 image."""
    vertices, faces, _normals = build_finger_mesh(geometry, sides=sides)
    centerline = geometry["centerline"]

    rotated = vertices @ _R.T
    xy = rotated[:, :2].copy()
    xy[:, 1] *= -1  # flip so "up" in 3D looks up on screen

    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = max((hi - lo).max(), 1e-6)
    scale = panel_size * (1.0 - 2 * margin_frac) / span
    center_px = panel_size / 2.0
    center_data = (lo + hi) / 2.0
    px = (xy - center_data) * scale + center_px

    panel = np.full((panel_size, panel_size, 3), 24, dtype=np.uint8)

    # tube rings + rounded tip-dome rings, in the same ring order build_finger_mesh
    # lays them out in - so the panel actually shows the rounded tip, not just the tube.
    n_ring_rows = len(centerline) + DEFAULT_TIP_CAP_RINGS
    ring_px = px[: n_ring_rows * sides].reshape(n_ring_rows, sides, 2)

    # tube "seam" lines (longitudinal) - every 4th vertex around the tube
    for k in range(0, sides, max(1, sides // 4)):
        pts = ring_px[:, k, :].astype(np.int32)
        cv2.polylines(panel, [pts], isClosed=False, color=(90, 90, 90), thickness=1, lineType=cv2.LINE_AA)

    # cross-section rings: every ~6th sample along the tube, but every ring
    # through the tip dome (only 6 of them) so the rounded cap actually reads
    # as rounded instead of vanishing between sparse cross-sections.
    n_tube_rings = len(centerline)
    cross_section_rows = list(range(0, n_tube_rings, 6)) + list(range(n_tube_rings, n_ring_rows))
    for i in cross_section_rows:
        pts = ring_px[i].astype(np.int32)
        cv2.polylines(panel, [pts], isClosed=True, color=(110, 70, 0), thickness=1, lineType=cv2.LINE_AA)

    # centerline itself, on top
    centerline_px = ((centerline @ _R.T)[:, :2] * [1, -1] - center_data) * scale + center_px
    cv2.polylines(panel, [centerline_px.astype(np.int32)], isClosed=False,
                  color=(0, 210, 255), thickness=2, lineType=cv2.LINE_AA)

    # MCP / TIP markers
    cv2.circle(panel, tuple(centerline_px[0].astype(int)), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(centerline_px[-1].astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "MCP", tuple(centerline_px[0].astype(int) + [8, -8]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "TIP", tuple(centerline_px[-1].astype(int) + [8, -8]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return panel
