"""
effects.py
==========
Low-level glow/bloom drawing primitives, plus the "spell" effect classes
(portal, projectile, lightning bolt, energy beam, shield throw).

Rendering strategy (cheap but effective "cinematic glow" without a GPU
shader pipeline):
  1. Everything magical is drawn onto a float32 BGR "glow layer" that starts
     at black, using additive draws (cv2 lines/circles/ellipses).
  2. We extract the bright regions of that layer, Gaussian-blur them a
     couple of times at increasing kernel size, and add the blurred result
     back on top of the sharp original (classic bloom).
  3. The bloomed glow layer is added (not alpha-blended) onto the camera
     frame, which is what gives the "glowing energy" look instead of a
     flat sticker look.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import BloomConfig, SpellConfig

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Glow layer primitives
# --------------------------------------------------------------------------- #
def new_glow_layer(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.float32)


def draw_glow_line(
    layer: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Color,
    thickness: int = 2,
    intensity: float = 1.0,
) -> None:
    """A line with a soft outer halo baked in via two passes."""
    pt1 = (int(round(p1[0])), int(round(p1[1])))
    pt2 = (int(round(p2[0])), int(round(p2[1])))
    c = tuple(float(v) * intensity for v in color)
    cv2.line(layer, pt1, pt2, tuple(v * 0.35 for v in c), thickness + 4, cv2.LINE_AA)
    cv2.line(layer, pt1, pt2, c, thickness, cv2.LINE_AA)


def draw_glow_circle(
    layer: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    color: Color,
    thickness: int = 2,
    intensity: float = 1.0,
    filled: bool = False,
) -> None:
    c = (int(round(center[0])), int(round(center[1])))
    r = max(int(round(radius)), 1)
    col = tuple(float(v) * intensity for v in color)
    if filled:
        cv2.circle(layer, c, r, tuple(v * 0.4 for v in col), -1, cv2.LINE_AA)
        cv2.circle(layer, c, max(r // 2, 1), col, -1, cv2.LINE_AA)
    else:
        cv2.circle(layer, c, r, tuple(v * 0.3 for v in col), thickness + 5, cv2.LINE_AA)
        cv2.circle(layer, c, r, col, thickness, cv2.LINE_AA)


def draw_glow_arc(
    layer: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    color: Color,
    thickness: int = 2,
    intensity: float = 1.0,
) -> None:
    c = (int(round(center[0])), int(round(center[1])))
    r = max(int(round(radius)), 1)
    col = tuple(float(v) * intensity for v in color)
    axes = (r, r)
    cv2.ellipse(layer, c, axes, 0, start_deg, end_deg, tuple(v * 0.3 for v in col), thickness + 4, cv2.LINE_AA)
    cv2.ellipse(layer, c, axes, 0, start_deg, end_deg, col, thickness, cv2.LINE_AA)


def draw_glow_polyline(
    layer: np.ndarray, pts: List[Tuple[float, float]], color: Color,
    thickness: int = 2, intensity: float = 1.0, closed: bool = False,
) -> None:
    arr = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
    if len(arr) < 2:
        return
    col = tuple(float(v) * intensity for v in color)
    cv2.polylines(layer, [arr], closed, tuple(v * 0.3 for v in col), thickness + 3, cv2.LINE_AA)
    cv2.polylines(layer, [arr], closed, col, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Bloom / compositing
# --------------------------------------------------------------------------- #
def apply_bloom(glow_layer: np.ndarray, cfg: BloomConfig) -> np.ndarray:
    """Threshold the bright pixels, blur them progressively, and add the
    blurred halo back on top of the sharp glow layer.
    """
    gray = cv2.cvtColor(np.clip(glow_layer, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)
    mask_f = mask.astype(np.float32) / 255.0
    bright = glow_layer * mask_f[..., None]

    k = cfg.blur_kernel | 1  # must be odd
    blurred = bright
    for i in range(cfg.passes):
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        k = (k * 2) | 1

    return glow_layer + blurred * cfg.intensity


def composite_additive(frame_bgr: np.ndarray, glow_layer: np.ndarray) -> np.ndarray:
    """Additively blend the (already bloomed) glow layer onto the camera
    frame. Additive blending is what makes overlapping glows blow out to
    white-hot the way fire/energy effects do in the movies.
    """
    base = frame_bgr.astype(np.float32)
    out = base + glow_layer
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Spell effects
# --------------------------------------------------------------------------- #
@dataclass
class Projectile:
    """A glowing bolt fired from a swipe gesture."""

    pos: List[float]
    vel: Tuple[float, float]
    born: float = field(default_factory=time.time)
    radius: float = 14.0
    color: Color = (60, 140, 255)
    trail: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, dt: float) -> None:
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.trail.append((self.pos[0], self.pos[1]))
        if len(self.trail) > 14:
            self.trail.pop(0)

    def alive(self, lifetime: float) -> bool:
        return (time.time() - self.born) < lifetime

    def draw(self, layer: np.ndarray) -> None:
        for i, (tx, ty) in enumerate(self.trail):
            fade = (i + 1) / max(len(self.trail), 1)
            draw_glow_circle(layer, (tx, ty), self.radius * 0.5 * fade, self.color, intensity=fade * 0.6, filled=True)
        draw_glow_circle(layer, tuple(self.pos), self.radius, self.color, intensity=1.2, filled=True)


@dataclass
class LightningBolt:
    """A short-lived jagged bolt between two points."""

    p1: Tuple[float, float]
    p2: Tuple[float, float]
    born: float = field(default_factory=time.time)
    color: Color = (255, 220, 120)
    segments: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._regenerate()

    def _regenerate(self) -> None:
        n = 12
        pts = []
        x1, y1 = self.p1
        x2, y2 = self.p2
        for i in range(n + 1):
            t = i / n
            bx = x1 + (x2 - x1) * t
            by = y1 + (y2 - y1) * t
            if 0 < i < n:
                jitter = (1 - abs(t - 0.5) * 2) * 18.0
                bx += np.random.uniform(-jitter, jitter)
                by += np.random.uniform(-jitter, jitter)
            pts.append((bx, by))
        self.segments = pts

    def alive(self, lifetime: float) -> bool:
        return (time.time() - self.born) < lifetime

    def draw(self, layer: np.ndarray) -> None:
        draw_glow_polyline(layer, self.segments, self.color, thickness=2, intensity=1.3)


@dataclass
class EnergyBeam:
    """A continuous glowing beam from a hand outward in a fixed direction."""

    origin: Tuple[float, float]
    direction: Tuple[float, float]
    born: float = field(default_factory=time.time)
    color: Color = (80, 160, 255)
    length: float = 700.0
    width: float = 10.0

    def alive(self, lifetime: float) -> bool:
        return (time.time() - self.born) < lifetime

    def draw(self, layer: np.ndarray) -> None:
        ox, oy = self.origin
        dx, dy = self.direction
        end = (ox + dx * self.length, oy + dy * self.length)
        age = time.time() - self.born
        flicker = 0.85 + 0.15 * math.sin(age * 40.0)
        draw_glow_line(layer, self.origin, end, self.color, thickness=int(self.width), intensity=flicker)
        draw_glow_circle(layer, self.origin, self.width * 1.4, self.color, intensity=flicker, filled=True)


@dataclass
class Portal:
    """Swirling portal effect anchored between two hands."""

    center: Tuple[float, float]
    radius: float
    rotation: float = 0.0
    color_a: Color = (60, 140, 255)
    color_b: Color = (200, 100, 255)

    def update(self, dt: float, speed_deg: float) -> None:
        self.rotation += math.radians(speed_deg) * dt

    def draw(self, layer: np.ndarray) -> None:
        cx, cy = self.center
        # Outer ring
        draw_glow_arc(layer, self.center, self.radius, math.degrees(self.rotation), math.degrees(self.rotation) + 300, self.color_a, thickness=4, intensity=1.0)
        # Inner counter-rotating ring
        draw_glow_arc(layer, self.center, self.radius * 0.7, -math.degrees(self.rotation) * 1.4, -math.degrees(self.rotation) * 1.4 + 260, self.color_b, thickness=3, intensity=0.9)
        # Swirling spiral arms for the "event horizon" feel
        for arm in range(3):
            pts = []
            arm_offset = arm * (2 * math.pi / 3)
            for i in range(24):
                t = i / 23
                ang = self.rotation * 2 + arm_offset + t * math.pi * 1.6
                rad = self.radius * (0.15 + 0.85 * t)
                pts.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))
            draw_glow_polyline(layer, pts, self.color_a, thickness=2, intensity=0.7 * (1 - arm * 0.2))
        # Bright core
        draw_glow_circle(layer, self.center, self.radius * 0.18, (255, 255, 255), intensity=0.8, filled=True)


@dataclass
class ShieldThrow:
    """Simple animation: the magic circle detaches and flies outward, then
    fades. Purely cosmetic bonus effect.
    """

    origin: Tuple[float, float]
    direction: Tuple[float, float]
    radius: float
    born: float = field(default_factory=time.time)
    speed: float = 650.0
    color: Color = (60, 140, 255)

    def alive(self, lifetime: float) -> bool:
        return (time.time() - self.born) < lifetime

    def position(self) -> Tuple[float, float]:
        age = time.time() - self.born
        return (
            self.origin[0] + self.direction[0] * self.speed * age,
            self.origin[1] + self.direction[1] * self.speed * age,
        )

    def draw(self, layer: np.ndarray, lifetime: float) -> None:
        age = time.time() - self.born
        fade = max(0.0, 1.0 - age / lifetime)
        pos = self.position()
        rot = age * 720.0  # spins fast while thrown
        for i in range(3):
            r = self.radius * (1.0 - i * 0.18)
            draw_glow_arc(layer, pos, r, rot + i * 40, rot + i * 40 + 280, self.color, thickness=2, intensity=fade)
