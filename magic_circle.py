"""
magic_circle.py
================
The centerpiece visual: a Doctor-Strange-style rotating magic circle that
anchors to a detected open palm.

Design:
  * Three concentric rings, each spinning at a different (and alternating)
    speed - this is what sells the "layered sigil" look rather than a
    single flat spinning disc.
  * A ring of procedurally drawn "rune" glyphs (small vector symbols, no
    external font/image assets needed) that rotate with the outer ring.
  * Radial spokes connecting the rings like a summoning circle.
  * A rotating geometric core symbol (triangle inscribed in hexagon).
  * Everything is drawn onto the shared glow layer via effects.py helpers
    so it automatically benefits from bloom + additive blending.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import MagicCircleConfig
import effects

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Procedural rune glyphs
# --------------------------------------------------------------------------- #
def _rune_triangle(layer: np.ndarray, center: Tuple[float, float], size: float, angle: float, color: Color, intensity: float) -> None:
    pts = []
    for k in range(3):
        a = angle + k * (2 * math.pi / 3)
        pts.append((center[0] + math.cos(a) * size, center[1] + math.sin(a) * size))
    effects.draw_glow_polyline(layer, pts, color, thickness=1, intensity=intensity, closed=True)


def _rune_cross(layer: np.ndarray, center: Tuple[float, float], size: float, angle: float, color: Color, intensity: float) -> None:
    for a_off in (0, math.pi / 2):
        a = angle + a_off
        p1 = (center[0] + math.cos(a) * size, center[1] + math.sin(a) * size)
        p2 = (center[0] - math.cos(a) * size, center[1] - math.sin(a) * size)
        effects.draw_glow_line(layer, p1, p2, color, thickness=1, intensity=intensity)


def _rune_eye(layer: np.ndarray, center: Tuple[float, float], size: float, angle: float, color: Color, intensity: float) -> None:
    effects.draw_glow_circle(layer, center, size * 0.55, color, thickness=1, intensity=intensity)
    dot = (center[0] + math.cos(angle) * size * 0.2, center[1] + math.sin(angle) * size * 0.2)
    effects.draw_glow_circle(layer, dot, size * 0.15, color, intensity=intensity, filled=True)


def _rune_zigzag(layer: np.ndarray, center: Tuple[float, float], size: float, angle: float, color: Color, intensity: float) -> None:
    pts = []
    n = 4
    for i in range(n + 1):
        t = i / n - 0.5
        perp = angle + math.pi / 2
        base = (center[0] + math.cos(angle) * t * size * 2, center[1] + math.sin(angle) * t * size * 2)
        off = size * 0.35 * (1 if i % 2 == 0 else -1)
        pts.append((base[0] + math.cos(perp) * off, base[1] + math.sin(perp) * off))
    effects.draw_glow_polyline(layer, pts, color, thickness=1, intensity=intensity)


_RUNE_LIBRARY = [_rune_triangle, _rune_cross, _rune_eye, _rune_zigzag]


# --------------------------------------------------------------------------- #
# Magic circle
# --------------------------------------------------------------------------- #
@dataclass
class MagicCircle:
    """A single magic circle instance, one per active shielding hand."""

    cfg: MagicCircleConfig
    center: Tuple[float, float] = (0.0, 0.0)
    target_radius: float = 0.0
    current_radius: float = 0.0
    ring_angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    manual_rotation: float = 0.0
    active: bool = False
    spawn_time: float = field(default_factory=time.time)
    _rune_choices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.target_radius = self.cfg.base_radius
        self.current_radius = self.cfg.base_radius * 0.05  # start small, "grow in"
        if not self._rune_choices:
            self._rune_choices = [i % len(_RUNE_LIBRARY) for i in range(self.cfg.rune_count)]

    def activate(self, center: Tuple[float, float]) -> None:
        if not self.active:
            self.spawn_time = time.time()
        self.active = True
        self.center = center

    def deactivate(self) -> None:
        self.active = False

    def set_center(self, center: Tuple[float, float]) -> None:
        self.center = center

    def set_radius_from_pinch(self, pinch01: float) -> None:
        """pinch01: 0 = pinched (small), 1 = open (large)."""
        self.target_radius = self.cfg.min_radius + pinch01 * (self.cfg.max_radius - self.cfg.min_radius)

    def add_manual_rotation(self, delta_rad: float) -> None:
        self.manual_rotation += delta_rad

    def update(self, dt: float) -> None:
        # smooth radius toward target (also drives the "grow in" spawn animation)
        s = self.cfg.radius_smoothing
        self.current_radius += (self.target_radius - self.current_radius) * min(1.0, s + dt)

        for i, speed in enumerate(self.cfg.ring_speeds_deg):
            self.ring_angles[i] += math.radians(speed) * dt

    def draw(self, layer: np.ndarray) -> None:
        if self.current_radius < 2:
            return

        cfg = self.cfg
        cx, cy = self.center
        r = self.current_radius
        base_angle = self.manual_rotation

        # --- outer segmented ring (dashed arcs, "sigil ring") -----------
        seg_count = 20
        gap_deg = 6
        seg_deg = (360.0 / seg_count) - gap_deg
        outer_deg_offset = math.degrees(self.ring_angles[0] + base_angle)
        for i in range(seg_count):
            start = outer_deg_offset + i * (360.0 / seg_count)
            effects.draw_glow_arc(
                layer, (cx, cy), r, start, start + seg_deg,
                cfg.color_mid, thickness=cfg.ring_thickness[0], intensity=cfg.glow_intensity,
            )

        # --- rune ring ----------------------------------------------------
        rune_r = r * 0.86
        rune_angle_offset = self.ring_angles[0] + base_angle
        for i in range(cfg.rune_count):
            a = rune_angle_offset + i * (2 * math.pi / cfg.rune_count)
            rc = (cx + math.cos(a) * rune_r, cy + math.sin(a) * rune_r)
            rune_fn = _RUNE_LIBRARY[self._rune_choices[i]]
            rune_fn(layer, rc, cfg.rune_size * (r / cfg.base_radius), a, cfg.color_spark, cfg.glow_intensity * 0.9)

        # --- middle solid ring --------------------------------------------
        mid_r = r * 0.68
        effects.draw_glow_circle(layer, (cx, cy), mid_r, cfg.color_core, thickness=cfg.ring_thickness[1], intensity=cfg.glow_intensity)

        # --- radial spokes (rotate with ring 2) ----------------------------
        spoke_angle_offset = self.ring_angles[1] + base_angle
        for i in range(cfg.radial_line_count):
            a = spoke_angle_offset + i * (2 * math.pi / cfg.radial_line_count)
            p1 = (cx + math.cos(a) * mid_r * 0.35, cy + math.sin(a) * mid_r * 0.35)
            p2 = (cx + math.cos(a) * mid_r, cy + math.sin(a) * mid_r)
            effects.draw_glow_line(layer, p1, p2, cfg.color_mid, thickness=1, intensity=cfg.glow_intensity * 0.8)

        # --- inner ring (fastest, opposite spin) ---------------------------
        inner_r = r * 0.42
        effects.draw_glow_circle(layer, (cx, cy), inner_r, cfg.color_edge, thickness=cfg.ring_thickness[2], intensity=cfg.glow_intensity)

        # --- core geometric symbol: hexagon + inscribed triangle -----------
        core_angle = self.ring_angles[2] + base_angle
        hexagon = []
        for i in range(6):
            a = core_angle + i * (2 * math.pi / 6)
            hexagon.append((cx + math.cos(a) * inner_r * 0.75, cy + math.sin(a) * inner_r * 0.75))
        effects.draw_glow_polyline(layer, hexagon, cfg.color_core, thickness=2, intensity=cfg.glow_intensity, closed=True)

        triangle = []
        for i in range(3):
            a = -core_angle * 1.3 + i * (2 * math.pi / 3)
            triangle.append((cx + math.cos(a) * inner_r * 0.55, cy + math.sin(a) * inner_r * 0.55))
        effects.draw_glow_polyline(layer, triangle, cfg.color_spark, thickness=2, intensity=cfg.glow_intensity, closed=True)

        # bright center point
        effects.draw_glow_circle(layer, (cx, cy), max(inner_r * 0.08, 3), (255, 255, 255), intensity=0.6, filled=True)
