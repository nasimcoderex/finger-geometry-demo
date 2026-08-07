"""camera -> MediaPipe hand detection -> finger-geometry ONNX model ->
geometry engine (reconstruct full profile) -> mesh -> live visualization.

This is a viewer, not a product: it exists to show what model.onnx actually
predicts on a live hand, side by side with the reconstructed 3D geometry.

Controls: q / Esc to quit, s to save the current mesh as mesh_TIMESTAMP.obj.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from geometry_engine import reconstruct_full_geometry
from hand_detector import HandDetector
from mesh import build_finger_mesh, save_obj
from predictor import GeometryPredictor
from ring import TemporalFilter, render_ring_overlay
from ring_model import load_real_diamondring
from viz import render_wireframe

ONNX_PATH = Path(__file__).parent / "model.onnx"
PANEL_SIZE = 420


def open_camera():
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_DSHOW
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"opened camera index {index}")
            return cap
        cap.release()
    raise RuntimeError("no camera found (tried indices 0,1,2)")


def draw_text_block(img, lines, origin=(10, 22), color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def draw_banner(img, text, color=(0, 200, 255)):
    """A prominent, centered message - the small corner status text is easy
    to miss, and this is meant to actively tell the user what to do."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    x, y = (w - tw) // 2, h - 30
    cv2.rectangle(img, (x - 12, y - th - 10), (x + tw + 12, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def main():
    if not ONNX_PATH.exists():
        sys.exit(f"model not found at {ONNX_PATH} - copy model.onnx here first")

    detector = HandDetector(max_hands=1)
    predictor = GeometryPredictor(ONNX_PATH)
    ring_mesh = load_real_diamondring(force_rebuild=True)  # GLB Ring stage: your diamondring.glb, decimated+recentered once
    ring_filter = TemporalFilter()  # Temporal Filter stage
    cap = open_camera()

    last_geometry = None
    fps_t0, fps_n, fps = time.time(), 0, 0.0

    print("running - press 'q' to quit, 's' to save the current mesh as OBJ")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("camera read failed, stopping")
                break
            frame_bgr = cv2.flip(frame_bgr, 1)  # mirror, feels natural for a front camera

            detection = detector.detect(frame_bgr)
            status_lines = [f"FPS {fps:.1f}"]

            if detection is not None:
                frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                centerline_mm, orientation_quat, finger_length_mm, _crop_box = predictor.predict(
                    frame_rgb_pil, detection["ring_finger_px"]
                )
                geometry = reconstruct_full_geometry(centerline_mm, orientation_quat, finger_length_mm)
                last_geometry = geometry

                clean_frame_bgr = frame_bgr.copy()  # pre-composite, for the ring's edge-measurement scan
                ring_status = render_ring_overlay(frame_bgr, geometry, detection, ring_mesh,
                                                  temporal_filter=ring_filter,
                                                  measurement_frame_bgr=clean_frame_bgr)
                if ring_status == "not_straight":
                    draw_banner(frame_bgr, "Straighten your ring finger to place the ring")
                elif ring_status == "fingers_together":
                    draw_banner(frame_bgr, "Spread your fingers apart to place the ring")
                status_lines += [
                    f"hand: {detection['handedness']}",
                    f"finger_length (model):  {finger_length_mm:6.1f} mm",
                    f"finger_length (curve):  {geometry['arc_length'][-1]:6.1f} mm",
                    f"radius MCP->TIP: {geometry['radius'][0]:.1f} -> {geometry['radius'][-1]:.1f} mm",
                ]
                panel = render_wireframe(geometry, panel_size=PANEL_SIZE)
            else:
                ring_filter.reset()  # avoid snapping/lerping from a stale pose once a hand reappears
                status_lines.append("no hand detected")
                panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 24, dtype=np.uint8)
                cv2.putText(panel, "no hand", (PANEL_SIZE // 2 - 50, PANEL_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)

            draw_text_block(frame_bgr, status_lines)

            cam_h, cam_w = frame_bgr.shape[:2]
            display_cam = cv2.resize(frame_bgr, (int(cam_w * PANEL_SIZE / cam_h), PANEL_SIZE))
            combined = np.hstack([display_cam, panel])
            cv2.imshow("finger-geometry-demo  (q=quit, s=save mesh)", combined)

            fps_n += 1
            if time.time() - fps_t0 >= 0.5:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_geometry is not None:
                vertices, faces, normals = build_finger_mesh(last_geometry)
                out_path = Path(__file__).parent / f"mesh_{int(time.time())}.obj"
                save_obj(out_path, vertices, faces, normals)
                print(f"saved {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
