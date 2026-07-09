"""
config.py
=========
Central place for every tunable parameter in the project.

Nothing in the other modules should hard-code a "magic number" that a user
might reasonably want to tweak (ring speed, colors, particle counts, gesture
thresholds, ...). Instead they should read it from here. This keeps the
effect modules clean and makes the whole app easy to re-theme or re-balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

Color = Tuple[int, int, int]  # BGR, OpenCV convention


# --------------------------------------------------------------------------- #
# Camera / capture
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    requested_fps: int = 30
    mirror: bool = True  # flip horizontally so it acts like a mirror


# --------------------------------------------------------------------------- #
# MediaPipe hand tracking
# --------------------------------------------------------------------------- #
@dataclass
class TrackingConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1  # 0 = fastest / lite, 1 = full
    # Exponential moving average factor for landmark smoothing.
    # Lower = smoother but laggier, higher = snappier but jittery.
    smoothing_alpha: float = 0.55


# --------------------------------------------------------------------------- #
# Gesture recognition thresholds
# --------------------------------------------------------------------------- #
@dataclass
class GestureConfig:
    # Fraction of fingers (of 4, excluding thumb) that must be extended
    # to count as an "open palm".
    open_palm_min_extended: int = 4
    fist_max_extended: int = 1

    # Pinch distance (pixels, normalized by hand span) below which we
    # consider thumb+index "pinched".
    pinch_close_ratio: float = 0.35
    pinch_open_ratio: float = 0.9

    # Wrist-rotation gesture sensitivity (radians -> circle rotation radians)
    rotation_gain: float = 1.4

    # Swipe detection: hand center must move at least this many pixels
    # per frame (roughly, normalized to 1280 width) in a consistent
    # direction over the trailing window to count as a swipe.
    swipe_history_len: int = 6
    swipe_min_speed_px: float = 55.0
    swipe_cooldown_seconds: float = 0.6

    # Two-hand portal distance change needed per frame to register as
    # "moving apart" / "moving together" (pixels).
    portal_distance_smoothing: float = 0.25


# --------------------------------------------------------------------------- #
# Magic circle visuals
# --------------------------------------------------------------------------- #
@dataclass
class MagicCircleConfig:
    base_radius: float = 120.0
    min_radius: float = 55.0
    max_radius: float = 260.0
    radius_smoothing: float = 0.18

    ring_count: int = 3
    # degrees / second for each ring (alternating direction looks best)
    ring_speeds_deg: Tuple[float, float, float] = (28.0, -40.0, 55.0)
    ring_thickness: Tuple[int, int, int] = (3, 2, 2)

    rune_count: int = 12
    rune_size: float = 14.0

    radial_line_count: int = 16

    glow_intensity: float = 0.9

    # Doctor-Strange-esque palette: burnt orange -> ember red -> warm gold
    color_core: Color = (60, 140, 255)     # bright orange (BGR)
    color_mid: Color = (30, 90, 235)       # deep orange-red
    color_edge: Color = (10, 40, 200)      # dark red
    color_spark: Color = (90, 200, 255)    # warm yellow-gold


# --------------------------------------------------------------------------- #
# Particle system
# --------------------------------------------------------------------------- #
@dataclass
class ParticleConfig:
    max_particles: int = 400
    spark_life_range: Tuple[float, float] = (0.35, 0.9)
    ember_life_range: Tuple[float, float] = (0.8, 1.8)
    trail_life_range: Tuple[float, float] = (0.2, 0.45)

    spark_speed_range: Tuple[float, float] = (40.0, 160.0)
    ember_speed_range: Tuple[float, float] = (10.0, 40.0)

    gravity: float = -18.0  # embers drift slightly "up" like fire, so negative
    drag: float = 0.985

    spark_size_range: Tuple[float, float] = (1.5, 3.5)
    ember_size_range: Tuple[float, float] = (1.0, 2.5)


# --------------------------------------------------------------------------- #
# Spell / bonus effects
# --------------------------------------------------------------------------- #
@dataclass
class SpellConfig:
    projectile_speed: float = 900.0       # px / second
    projectile_radius: float = 14.0
    projectile_lifetime: float = 1.4

    lightning_segments: int = 14
    lightning_jitter: float = 18.0
    lightning_lifetime: float = 0.25

    beam_lifetime: float = 0.6
    beam_width: float = 10.0

    portal_min_radius: float = 40.0
    portal_max_radius: float = 220.0
    portal_ring_speed_deg: float = 70.0


# --------------------------------------------------------------------------- #
# Bloom / post-processing
# --------------------------------------------------------------------------- #
@dataclass
class BloomConfig:
    threshold: int = 150
    blur_kernel: int = 25
    intensity: float = 0.55
    passes: int = 2


# --------------------------------------------------------------------------- #
# Aggregate config
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    circle: MagicCircleConfig = field(default_factory=MagicCircleConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    spell: SpellConfig = field(default_factory=SpellConfig)
    bloom: BloomConfig = field(default_factory=BloomConfig)

    window_name: str = "Doctor Strange Magic Circle"
    screenshot_dir: str = "captures"
    recording_dir: str = "captures"


CONFIG = AppConfig()
