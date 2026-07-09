"""
particles.py
============
A small, dependency-free particle system for fire sparks, floating embers,
and hand-motion energy trails. All particles fade out over their lifetime
and are drawn additively via effects.draw_glow_circle, so dense clusters
naturally bloom into bright hot-spots.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import ParticleConfig
import effects

Color = Tuple[int, int, int]


@dataclass
class Particle:
    pos: List[float]
    vel: List[float]
    born: float
    lifetime: float
    size: float
    color: Color
    kind: str  # "spark" | "ember" | "trail"

    def age(self) -> float:
        return time.time() - self.born

    def alive(self) -> bool:
        return self.age() < self.lifetime

    def fade(self) -> float:
        return max(0.0, 1.0 - self.age() / self.lifetime)


class ParticleSystem:
    """Owns and updates every particle in the scene."""

    def __init__(self, cfg: ParticleConfig):
        self.cfg = cfg
        self._particles: List[Particle] = []

    # --------------------------------------------------------------- #
    # Spawning
    # --------------------------------------------------------------- #
    def spawn_spark(self, pos: Tuple[float, float], color: Color, count: int = 1) -> None:
        cfg = self.cfg
        for _ in range(count):
            if len(self._particles) >= cfg.max_particles:
                return
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(*cfg.spark_speed_range)
            self._particles.append(
                Particle(
                    pos=[pos[0], pos[1]],
                    vel=[np.cos(angle) * speed, np.sin(angle) * speed],
                    born=time.time(),
                    lifetime=random.uniform(*cfg.spark_life_range),
                    size=random.uniform(*cfg.spark_size_range),
                    color=color,
                    kind="spark",
                )
            )

    def spawn_ember(self, pos: Tuple[float, float], color: Color, count: int = 1) -> None:
        cfg = self.cfg
        for _ in range(count):
            if len(self._particles) >= cfg.max_particles:
                return
            angle = random.uniform(-np.pi, np.pi)
            speed = random.uniform(*cfg.ember_speed_range)
            self._particles.append(
                Particle(
                    pos=[pos[0] + random.uniform(-6, 6), pos[1] + random.uniform(-6, 6)],
                    vel=[np.cos(angle) * speed * 0.3, cfg.gravity + np.sin(angle) * speed * 0.3],
                    born=time.time(),
                    lifetime=random.uniform(*cfg.ember_life_range),
                    size=random.uniform(*cfg.ember_size_range),
                    color=color,
                    kind="ember",
                )
            )

    def spawn_trail(self, pos: Tuple[float, float], vel: Tuple[float, float], color: Color) -> None:
        cfg = self.cfg
        if len(self._particles) >= cfg.max_particles:
            return
        self._particles.append(
            Particle(
                pos=[pos[0], pos[1]],
                vel=[-vel[0] * 0.15, -vel[1] * 0.15],
                born=time.time(),
                lifetime=random.uniform(*cfg.trail_life_range),
                size=random.uniform(2.0, 4.0),
                color=color,
                kind="trail",
            )
        )

    # --------------------------------------------------------------- #
    # Update / draw
    # --------------------------------------------------------------- #
    def update(self, dt: float) -> None:
        cfg = self.cfg
        alive: List[Particle] = []
        for p in self._particles:
            if not p.alive():
                continue
            p.vel[0] *= cfg.drag
            p.vel[1] = p.vel[1] * cfg.drag + (cfg.gravity * dt if p.kind == "ember" else 0.0)
            p.pos[0] += p.vel[0] * dt
            p.pos[1] += p.vel[1] * dt
            alive.append(p)
        self._particles = alive

    def draw(self, layer: np.ndarray) -> None:
        for p in self._particles:
            fade = p.fade()
            if fade <= 0:
                continue
            effects.draw_glow_circle(
                layer, tuple(p.pos), p.size, p.color, intensity=fade, filled=True
            )

    def count(self) -> int:
        return len(self._particles)
