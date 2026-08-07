"""Wraps the ONNX model: takes a full camera frame + the 4 ring-finger pixel
landmarks from hand_detector.py, reproduces the exact crop/preprocessing
F:\\finger-geometry-model's training/dataset.py used at training time
(margin 2.0x bbox, no augmentation jitter - this is inference, not training),
and runs the model.

Model I/O (from that repo's training/model.py / tools/export_onnx.py):
  input  "image"          (1,3,256,256) float32, ImageNet-normalized
  output "centerline"     (1,50,3) mm, MCP-relative
  output "orientation"    (1,4) unit quaternion (w,x,y,z)
  output "finger_length"  (1,) mm
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

INPUT_IMAGE_SIZE = 256
CROP_MARGIN_FACTOR = 2.0      # matches training/dataset.py's CROP_MARGIN_FACTOR
MIN_CROP_SIZE_PX = 20

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def compute_crop_box(ring_finger_px, margin_factor=CROP_MARGIN_FACTOR):
    pts = np.asarray(ring_finger_px)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    box_size = max(x1 - x0, y1 - y0, MIN_CROP_SIZE_PX)
    half = box_size * margin_factor / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


class GeometryPredictor:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def predict(self, frame_rgb_pil, ring_finger_px):
        """frame_rgb_pil: PIL Image (RGB), the full camera frame.
        ring_finger_px: (4,2) array [MCP,PIP,DIP,TIP] pixel coords in that frame.

        Returns (centerline_mm (50,3), orientation_quat (4,), finger_length_mm
        float, crop_box (x0,y0,x1,y1) - the last one so callers can draw it).
        """
        box = compute_crop_box(ring_finger_px)
        crop = frame_rgb_pil.crop(box).resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), Image.BILINEAR)

        arr = np.asarray(crop).astype(np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        image = arr[None, ...]  # (1,3,256,256)

        centerline, orientation, length = self.session.run(
            ["centerline", "orientation", "finger_length"], {"image": image}
        )
        return centerline[0], orientation[0], float(length[0]), box
