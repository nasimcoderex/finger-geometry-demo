"""Wraps MediaPipe's Tasks-API HandLandmarker (the only hand-landmark API
mediapipe>=1.0 ships - the older mp.solutions.hands was removed in this
version) and extracts the 4 ring-finger landmarks the ONNX model's crop box
needs.

Landmark topology (unchanged between the old and new API): 21 points,
0=wrist, 5/9/13/17=index/middle/ring/pinky MCP, each finger MCP->PIP->DIP->TIP
in order. Ring finger = indices [13,14,15,16] (MCP,PIP,DIP,TIP) - the SAME
convention F:\\finger-geometry-model's label_generator/ring_finger.py uses
for its freihand/ho3d MANO-order mapping, so no reindexing is needed to feed
this detector's output into that model's crop logic.
"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

RING_FINGER_INDICES = [13, 14, 15, 16]  # MCP, PIP, DIP, TIP


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmark model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("done")
    return MODEL_PATH


class HandDetector:
    def __init__(self, max_hands=1, min_detection_confidence=0.6):
        model_path = ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame_bgr):
        """frame_bgr: HxWx3 uint8 (as read by cv2.VideoCapture).

        Returns None if no hand found, else a dict:
          landmarks_px: (21,2) float32 pixel coords in this frame
          ring_finger_px: (4,2) float32 pixel coords [MCP,PIP,DIP,TIP]
          handedness: "Left" or "Right" (as MediaPipe sees the image, i.e.
                      mirrored vs. the subject's own left/right for a
                      front-facing selfie-style camera)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]  # first detected hand only
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        handedness = result.handedness[0][0].category_name if result.handedness else "?"

        return {
            "landmarks_px": pts,
            "ring_finger_px": pts[RING_FINGER_INDICES],
            "handedness": handedness,
        }

    def close(self):
        self._landmarker.close()
