"""
gestures.py
===========
Turns raw hand landmarks into semantic gesture state:

  * open_palm / fist            -> shield activate / deactivate
  * pinch strength [0..1]       -> shield resize
  * wrist rotation delta        -> manual magic-circle rotation
  * two-hand distance + delta   -> portal size / grow / shrink
  * swipe events                -> "shoot projectile" trigger

Nothing here touches OpenCV drawing; this module is pure geometry/state so
it's easy to unit test independently of rendering.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from config import GestureConfig
from hand_tracker import (
    Hand,
    WRIST,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    RING_MCP,
    RING_TIP,
    PINKY_MCP,
    PINKY_TIP,
)


# --------------------------------------------------------------------------- #
# Low level finger-state helpers
# --------------------------------------------------------------------------- #
def _finger_extended(hand: Hand, mcp_idx: int, tip_idx: int) -> bool:
    """A finger counts as 'extended' if its tip is meaningfully farther
    from the wrist than its MCP knuckle is. This is orientation-agnostic
    (works whether the hand is upright, sideways, or upside down) which a
    naive "tip.y < pip.y" check is not.
    """
    wx, wy = hand.point(WRIST)
    mx, my = hand.point(mcp_idx)
    tx, ty = hand.point(tip_idx)
    d_mcp = math.hypot(mx - wx, my - wy)
    d_tip = math.hypot(tx - wx, ty - wy)
    return d_tip > d_mcp * 1.15


def _count_extended_fingers(hand: Hand) -> int:
    fingers = [
        (INDEX_MCP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_TIP),
        (RING_MCP, RING_TIP),
        (PINKY_MCP, PINKY_TIP),
    ]
    return sum(1 for mcp, tip in fingers if _finger_extended(hand, mcp, tip))


def is_open_palm(hand: Hand, cfg: GestureConfig) -> bool:
    return _count_extended_fingers(hand) >= cfg.open_palm_min_extended


def is_fist(hand: Hand, cfg: GestureConfig) -> bool:
    return _count_extended_fingers(hand) <= cfg.fist_max_extended


def pinch_strength(hand: Hand, cfg: GestureConfig) -> float:
    """0.0 = fully pinched (thumb+index touching), 1.0 = fully open.
    Normalized against hand span so it's roughly distance-to-camera
    invariant.
    """
    tx, ty = hand.point(THUMB_TIP)
    ix, iy = hand.point(INDEX_TIP)
    dist = math.hypot(tx - ix, ty - iy)
    span = hand.span()
    ratio = dist / span
    lo, hi = cfg.pinch_close_ratio, cfg.pinch_open_ratio
    norm = (ratio - lo) / max(hi - lo, 1e-6)
    return max(0.0, min(1.0, norm))


def wrist_orientation_angle(hand: Hand) -> float:
    """Angle (radians) of the vector wrist -> middle_mcp. Used as a stable
    'compass heading' for the hand so we can measure rotation over time.
    """
    wx, wy = hand.point(WRIST)
    mx, my = hand.point(MIDDLE_MCP)
    return math.atan2(my - wy, mx - wx)


def _shortest_angle_delta(a_new: float, a_old: float) -> float:
    """Delta between two angles, wrapped to [-pi, pi] so a rotation crossing
    the +/-pi boundary doesn't produce a huge spurious jump.
    """
    d = a_new - a_old
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


@dataclass
class SwipeEvent:
    hand_key: str
    direction: Tuple[float, float]  # unit vector
    origin: Tuple[float, float]
    speed: float
    timestamp: float


@dataclass
class _HandHistory:
    positions: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=12)
    )  # (x, y, t)
    last_angle: Optional[float] = None
    last_swipe_time: float = 0.0


@dataclass
class HandGestureState:
    """Per-hand computed gesture snapshot for a single frame."""

    hand_key: str
    handedness: str
    open_palm: bool
    fist: bool
    pinch: float
    rotation_delta: float
    center: Tuple[float, float]
    span: float


class GestureRecognizer:
    """Stateful gesture recognizer. Call `update(hands)` once per frame."""

    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self._history: Dict[str, _HandHistory] = {}
        self._prev_portal_distance: Optional[float] = None

    def update(
        self, hands: List[Hand]
    ) -> Tuple[Dict[str, HandGestureState], List[SwipeEvent], Optional[float]]:
        """
        Returns:
            per_hand: dict keyed by "Left"/"Right" (or "Left_1" if dupes)
            swipes: list of SwipeEvent detected this frame
            portal_distance_delta: change in inter-hand distance if two
                hands are present (positive = moving apart), else None.
        """
        now = time.time()
        per_hand: Dict[str, HandGestureState] = {}
        swipes: List[SwipeEvent] = []
        active_keys = set()

        for i, hand in enumerate(hands):
            key = f"{hand.handedness}_{i}"
            active_keys.add(key)
            hist = self._history.setdefault(key, _HandHistory())

            cx, cy = hand.center
            hist.positions.append((cx, cy, now))

            angle = wrist_orientation_angle(hand)
            rot_delta = 0.0
            if hist.last_angle is not None:
                rot_delta = _shortest_angle_delta(angle, hist.last_angle) * self.cfg.rotation_gain
            hist.last_angle = angle

            state = HandGestureState(
                hand_key=key,
                handedness=hand.handedness,
                open_palm=is_open_palm(hand, self.cfg),
                fist=is_fist(hand, self.cfg),
                pinch=pinch_strength(hand, self.cfg),
                rotation_delta=rot_delta,
                center=(cx, cy),
                span=hand.span(),
            )
            per_hand[key] = state

            swipe = self._detect_swipe(key, hist, now)
            if swipe is not None:
                swipes.append(swipe)

        # forget hands no longer visible
        for stale_key in list(self._history.keys()):
            if stale_key not in active_keys:
                del self._history[stale_key]

        portal_delta = None
        if len(hands) == 2 and all(is_open_palm(h, self.cfg) for h in hands):
            d = math.hypot(
                hands[0].center[0] - hands[1].center[0],
                hands[0].center[1] - hands[1].center[1],
            )
            if self._prev_portal_distance is not None:
                portal_delta = d - self._prev_portal_distance
            self._prev_portal_distance = d
        else:
            self._prev_portal_distance = None

        return per_hand, swipes, portal_delta

    def _detect_swipe(
        self, key: str, hist: _HandHistory, now: float
    ) -> Optional[SwipeEvent]:
        cfg = self.cfg
        if len(hist.positions) < cfg.swipe_history_len:
            return None
        if now - hist.last_swipe_time < cfg.swipe_cooldown_seconds:
            return None

        pts = list(hist.positions)[-cfg.swipe_history_len :]
        (x0, y0, t0), (x1, y1, t1) = pts[0], pts[-1]
        dt = max(t1 - t0, 1e-3)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        speed = math.hypot(vx, vy)

        # normalize speed measurement against elapsed frames roughly
        if speed < cfg.swipe_min_speed_px * cfg.swipe_history_len:
            return None

        # require reasonably consistent direction (not just jitter):
        # compare displacement vector to a straight-line distance check.
        straight_dist = math.hypot(x1 - x0, y1 - y0)
        path_dist = sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )
        if path_dist < 1e-3 or straight_dist / path_dist < 0.6:
            return None

        hist.last_swipe_time = now
        norm = math.hypot(vx, vy) + 1e-6
        return SwipeEvent(
            hand_key=key,
            direction=(vx / norm, vy / norm),
            origin=(x1, y1),
            speed=speed,
            timestamp=now,
        )
