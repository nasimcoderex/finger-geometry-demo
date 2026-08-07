"""Exercises every module without a camera or a real hand in frame: ONNX
inference on a synthetic crop, geometry reconstruction, mesh building, and
the MediaPipe model download + a no-hand detect() call. Run once after
setup to catch import/shape/API errors before plugging in a live camera."""
import time
from pathlib import Path

import numpy as np
from PIL import Image

from geometry_engine import reconstruct_full_geometry
from mesh import build_finger_mesh, save_obj
from predictor import GeometryPredictor
from viz import render_wireframe

ONNX_PATH = Path(__file__).parent / "model.onnx"

print("1. loading ONNX model...")
predictor = GeometryPredictor(ONNX_PATH)
print("   OK")

print("2. running inference on a synthetic 480x640 frame + fake ring-finger points...")
frame = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
ring_finger_px = np.array([[300, 200], [305, 170], [308, 145], [310, 120]], dtype=np.float32)
centerline_mm, orientation_quat, finger_length_mm, crop_box = predictor.predict(frame, ring_finger_px)
print(f"   centerline {centerline_mm.shape}  orientation {orientation_quat} "
      f"(norm={np.linalg.norm(orientation_quat):.4f})  finger_length {finger_length_mm:.1f}mm  "
      f"crop_box {tuple(round(v, 1) for v in crop_box)}")

print("3. reconstructing full geometry...")
geometry = reconstruct_full_geometry(centerline_mm, orientation_quat, finger_length_mm)
print(f"   arc_length[-1]={geometry['arc_length'][-1]:.1f}mm  "
      f"radius[0]={geometry['radius'][0]:.2f}mm  radius[-1]={geometry['radius'][-1]:.2f}mm")
assert geometry["centerline"].shape == (50, 3)
assert geometry["tangent"].shape == (50, 3)
tangent_norms = np.linalg.norm(geometry["tangent"], axis=1)
assert np.allclose(tangent_norms, 1.0, atol=1e-4), tangent_norms
dots = np.sum(geometry["tangent"] * geometry["normal"], axis=1)
assert np.allclose(dots, 0.0, atol=1e-4), dots
print("   OK (unit tangents, tangent perp normal)")

print("4. building mesh...")
vertices, faces, normals = build_finger_mesh(geometry, sides=16)
print(f"   vertices {vertices.shape}  faces {faces.shape}  normals {normals.shape}")
assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
out_obj = Path(__file__).parent / "smoke_test_mesh.obj"
save_obj(out_obj, vertices, faces, normals)
print(f"   wrote {out_obj} ({out_obj.stat().st_size} bytes)")

print("5. rendering wireframe panel...")
panel = render_wireframe(geometry)
print(f"   panel shape {panel.shape}, non-background pixels: {(panel != 24).any(axis=-1).sum()}")

print("6. MediaPipe model download + no-hand detect()...")
from hand_detector import HandDetector  # noqa: E402  (import here: slow, only needed for this check)
t0 = time.time()
detector = HandDetector()
print(f"   model ready in {time.time()-t0:.1f}s")
blank = np.zeros((480, 640, 3), dtype=np.uint8)
result = detector.detect(blank)
print(f"   detect() on blank frame -> {result} (expected None)")
assert result is None
detector.close()

print("\nALL SMOKE TESTS PASSED")
