"""
hand_tracker.py
===============
Thin, well-behaved wrapper around MediaPipe's Hands solution.

Responsibilities:
  * Run MediaPipe inference on a BGR frame.
  * Convert normalized landmarks to pixel space (and keep normalized z).
  * Apply per-landmark exponential smoothing so the magic circle doesn't
    jitter even though raw landmark detection is noisy frame to frame.
  * Hand back a small, clean `Hand` dataclass instead of leaking MediaPipe's
    protobuf types into the rest of the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from config import TrackingConfig

# MediaPipe landmark index reference (used throughout gestures.py too):
#  0 WRIST
#  1-4   THUMB  (CMC, MCP, IP, TIP)
#  5-8   INDEX  (MCP, PIP, DIP, TIP)
#  9-12  MIDDLE (MCP, PIP, DIP, TIP)
#  13-16 RING   (MCP, PIP, DIP, TIP)
#  17-20 PINKY  (MCP, PIP, DIP, TIP)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20


@dataclass
class Hand:
    """A single tracked hand, landmarks already in pixel coordinates."""

    landmarks: np.ndarray  # shape (21, 3) -> x_px, y_px, z_normalized
    handedness: str        # "Left" or "Right"
    score: float
    center: Tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        self.center = (
            float(np.mean(self.landmarks[:, 0])),
            float(np.mean(self.landmarks[:, 1])),
        )

    def point(self, idx: int) -> Tuple[float, float]:
        """Pixel-space (x, y) for a given landmark index."""
        return float(self.landmarks[idx, 0]), float(self.landmarks[idx, 1])

    def span(self) -> float:
        """A rough scale reference for this hand (wrist -> middle MCP)."""
        wx, wy = self.point(WRIST)
        mx, my = self.point(MIDDLE_MCP)
        return float(np.hypot(mx - wx, my - wy)) + 1e-6


class _SmoothingBuffer:
    """Per-hand exponential moving average over landmark arrays."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._state: Dict[str, np.ndarray] = {}

    def smooth(self, key: str, landmarks: np.ndarray) -> np.ndarray:
        prev = self._state.get(key)
        if prev is None or prev.shape != landmarks.shape:
            self._state[key] = landmarks.copy()
            return landmarks
        blended = self.alpha * landmarks + (1.0 - self.alpha) * prev
        self._state[key] = blended
        return blended

    def forget_stale(self, active_keys: set) -> None:
        for key in list(self._state.keys()):
            if key not in active_keys:
                del self._state[key]


class HandTracker:
    """Runs MediaPipe Hands on frames and returns smoothed `Hand` objects."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            model_complexity=cfg.model_complexity,
        )
        self._smoother = _SmoothingBuffer(cfg.smoothing_alpha)

    def process(self, frame_bgr: np.ndarray) -> List[Hand]:
        """Run detection on one frame, return a list of smoothed hands."""
        h, w = frame_bgr.shape[:2]

        # MediaPipe wants RGB; mark not-writeable for a small perf win
        # (avoids an internal copy since we're not mutating it).
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        hands: List[Hand] = []
        active_keys: set = set()

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label, score = "Right", 1.0
                if handedness_list and i < len(handedness_list):
                    cls = handedness_list[i].classification[0]
                    label, score = cls.label, cls.score

                pts = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                # Key by handedness + slot so two hands don't smear together
                # if MediaPipe swaps ordering between frames.
                key = f"{label}_{i}"
                pts = self._smoother.smooth(key, pts)
                active_keys.add(key)
                hands.append(Hand(landmarks=pts, handedness=label, score=score))

        self._smoother.forget_stale(active_keys)
        return hands

    def close(self) -> None:
        self._hands.close()


def draw_debug_landmarks(frame: np.ndarray, hand: Hand, color=(0, 255, 180)) -> None:
    """Draw all 21 landmarks + basic skeleton connections for debugging."""
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring
        (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
        (5, 9), (9, 13), (13, 17),               # palm
    ]
    pts = hand.landmarks[:, :2].astype(int)
    for a, b in connections:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
