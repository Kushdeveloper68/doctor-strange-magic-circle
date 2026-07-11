"""
main.py
=======
Doctor Strange Magic Circle - application entry point.

Run with:  python main.py

Controls
--------
  M    Toggle magic-circle rendering
  P    Toggle portal mode (two open hands)
  D    Toggle debug landmark overlay
  F    Toggle fullscreen
  1/2/3  Select spell type (Projectile / Lightning / Beam) fired on swipe
  C    Save a screenshot
  V    Start/stop video recording
  ESC  Exit

Gestures
--------
  Open palm            -> spawn/activate a magic circle at that palm
  Closed fist           -> deactivate that hand's magic circle
  Pinch (thumb+index)   -> resize the active circle
  Rotate wrist           -> manually rotate the circle
  Two open hands         -> portal effect between the hands
  Hands apart/together   -> grow/shrink the portal
  Swipe                  -> fire the selected spell
  Draw a circle in the air with your index finger (either hand)
                          -> opens a "Sling Ring" portal at that spot,
                             which then follows your fingertip around.
                             Draw another circle, or make a fist, to
                             close it.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import CONFIG, AppConfig
from hand_tracker import HandTracker, Hand, draw_debug_landmarks, INDEX_TIP
from gestures import GestureRecognizer, HandGestureState, SwipeEvent, CircleEvent
from magic_circle import MagicCircle
from particles import ParticleSystem
import effects


SPELL_NAMES = ["Projectile", "Lightning", "Beam"]


class FPSCounter:
    """Simple smoothed FPS counter using an exponential moving average."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._fps = 0.0
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = max(now - self._last, 1e-6)
        self._last = now
        inst_fps = 1.0 / dt
        self._fps = self.smoothing * self._fps + (1 - self.smoothing) * inst_fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class DoctorStrangeApp:
    """Owns the capture loop, all subsystems, and UI/keyboard handling."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tracker = HandTracker(cfg.tracking)
        self.gestures = GestureRecognizer(cfg.gesture, cfg.circle_trace)
        self.particles = ParticleSystem(cfg.particles)

        self.circles: Dict[str, MagicCircle] = {}
        self.portal: Optional[effects.Portal] = None
        # portals opened by tracing a circle with an index fingertip,
        # keyed by hand_key so each hand can carry at most one.
        self.traced_portals: Dict[str, effects.Portal] = {}
        self.projectiles: List[effects.Projectile] = []
        self.lightning_bolts: List[effects.LightningBolt] = []
        self.beams: List[effects.EnergyBeam] = []
        self.shield_throws: List[effects.ShieldThrow] = []

        # feature toggles
        self.magic_mode = True
        self.portal_mode = True
        self.debug_mode = False
        self.fullscreen = False
        self.spell_index = 0

        # persistence buffer for a cheap motion-blur / energy-trail look
        self._glow_persist: Optional[np.ndarray] = None
        self._persist_decay = 0.78

        self.fps_counter = FPSCounter()
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._last_time = time.time()

        os.makedirs(cfg.screenshot_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(cfg.camera.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.camera.requested_fps)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that a camera is connected and "
                "not in use by another application."
            )

        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Failed to read frame from webcam.")
                    break

                if self.cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.time()
                dt = min(now - self._last_time, 0.05)  # clamp to avoid huge jumps
                self._last_time = now

                frame = self._process_frame(frame, dt)

                fps = self.fps_counter.tick()
                self._draw_ui(frame, fps)

                if self._video_writer is not None:
                    self._video_writer.write(frame)

                cv2.imshow(self.cfg.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame):
                    break
        finally:
            self._cleanup()

    # ------------------------------------------------------------------ #
    # Per-frame processing
    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray, dt: float) -> np.ndarray:
        h, w = frame.shape[:2]
        hands = self.tracker.process(frame)
        per_hand, swipes, portal_delta, circle_events = self.gestures.update(hands)
        hands_by_key = {f"{hand.handedness}_{i}": hand for i, hand in enumerate(hands)}

        glow_layer = effects.new_glow_layer((h, w))

        two_open = self.portal_mode and len(hands) == 2 and all(
            s.open_palm for s in per_hand.values()
        )

        if two_open:
            self._update_portal(hands, per_hand, dt, portal_delta)
            # while the two-hand portal is active, personal circles are
            # suppressed so the two effects don't visually compete.
            for c in self.circles.values():
                c.deactivate()
        else:
            self.portal = None
            self._update_circles(hands, per_hand, dt)

        self._handle_swipes(swipes, hands, per_hand)
        self._handle_circle_events(circle_events)
        self._update_traced_portals(hands_by_key, per_hand, dt)
        self._update_spells(dt)
        self._spawn_ambient_particles(hands, per_hand, dt)
        self.particles.update(dt)

        # ---- draw everything onto the glow layer ----
        if self.magic_mode:
            for circle in self.circles.values():
                if circle.active:
                    circle.draw(glow_layer)
        if self.portal is not None:
            self.portal.draw(glow_layer)
        for tp in self.traced_portals.values():
            tp.draw(glow_layer)
        for p in self.projectiles:
            p.draw(glow_layer)
        for b in self.lightning_bolts:
            b.draw(glow_layer)
        for beam in self.beams:
            beam.draw(glow_layer)
        for st in self.shield_throws:
            st.draw(glow_layer, self.cfg.spell.beam_lifetime + 0.4)
        self.particles.draw(glow_layer)

        if self.debug_mode:
            for hand in hands:
                draw_debug_landmarks(frame, hand)

        # persistence buffer -> cheap motion blur / trailing glow
        if self._glow_persist is None or self._glow_persist.shape != glow_layer.shape:
            self._glow_persist = glow_layer.copy()
        else:
            self._glow_persist = self._glow_persist * self._persist_decay + glow_layer
        combined = self._glow_persist

        bloomed = effects.apply_bloom(combined, self.cfg.bloom)
        frame = effects.composite_additive(frame, bloomed)

        self._current_hand_states = per_hand  # for UI status text
        self._current_hand_count = len(hands)
        return frame

    # ------------------------------------------------------------------ #
    # Magic circle management
    # ------------------------------------------------------------------ #
    def _update_circles(self, hands: List[Hand], per_hand: Dict[str, HandGestureState], dt: float) -> None:
        active_keys = set()
        for i, hand in enumerate(hands):
            key = f"{hand.handedness}_{i}"
            active_keys.add(key)
            state = per_hand.get(key)
            if state is None:
                continue

            circle = self.circles.get(key)
            if circle is None:
                circle = MagicCircle(self.cfg.circle)
                self.circles[key] = circle

            if state.open_palm:
                circle.activate(state.center)
                circle.set_center(state.center)
                circle.set_radius_from_pinch(state.pinch)
            elif state.fist:
                circle.deactivate()
            else:
                # neither a clean fist nor open palm: keep tracking position
                # if it was already active, so the circle doesn't "snap"
                # away during transitional hand poses.
                if circle.active:
                    circle.set_center(state.center)

            circle.add_manual_rotation(state.rotation_delta)
            circle.update(dt)

        # keep circles for hands that briefly vanished from tracking, but
        # drop ones that have been gone a while by simply leaving them be;
        # MediaPipe re-acquires the same key most of the time. Uncomment to
        # hard-remove if desired:
        # for stale_key in list(self.circles.keys()):
        #     if stale_key not in active_keys:
        #         del self.circles[stale_key]

    # ------------------------------------------------------------------ #
    # Portal management
    # ------------------------------------------------------------------ #
    def _update_portal(
        self,
        hands: List[Hand],
        per_hand: Dict[str, HandGestureState],
        dt: float,
        portal_delta: Optional[float],
    ) -> None:
        cx = (hands[0].center[0] + hands[1].center[0]) / 2.0
        cy = (hands[0].center[1] + hands[1].center[1]) / 2.0
        spell_cfg = self.cfg.spell

        if self.portal is None:
            dist = math.hypot(hands[0].center[0] - hands[1].center[0], hands[0].center[1] - hands[1].center[1])
            initial_r = float(np.clip(dist * 0.4, spell_cfg.portal_min_radius, spell_cfg.portal_max_radius))
            self.portal = effects.Portal(center=(cx, cy), radius=initial_r)
        else:
            self.portal.center = (cx, cy)
            if portal_delta is not None:
                new_r = self.portal.radius + portal_delta * self.cfg.gesture.portal_distance_smoothing
                self.portal.radius = float(np.clip(new_r, spell_cfg.portal_min_radius, spell_cfg.portal_max_radius))

        self.portal.update(dt, spell_cfg.portal_ring_speed_deg)

    # ------------------------------------------------------------------ #
    # Finger-drawn "Sling Ring" portals
    # ------------------------------------------------------------------ #
    def _handle_circle_events(self, circle_events: List[CircleEvent]) -> None:
        """A hand just finished tracing a loop with its index finger.
        First trace on a given hand opens a portal there; tracing again
        on a hand that already has one open acts as a toggle and closes
        it (mirrors how re-drawing feels natural rather than stacking
        portals on the same hand).
        """
        for ev in circle_events:
            existing = self.traced_portals.get(ev.hand_key)
            if existing is not None and existing.state in ("opening", "open"):
                existing.begin_close()
                continue

            portal = effects.Portal(
                center=ev.center,
                radius=float(np.clip(ev.radius * 1.8, self.cfg.spell.portal_min_radius, self.cfg.spell.portal_max_radius)),
                current_radius=1.0,
                color_a=self.cfg.circle.color_core,
                color_b=self.cfg.circle.color_spark,
                state="opening",
                anchor_hand_key=ev.hand_key,
            )
            self.traced_portals[ev.hand_key] = portal
            # a little burst of sparks at the moment of opening sells the effect
            self.particles.spawn_spark(ev.center, self.cfg.circle.color_spark, count=14)

    def _update_traced_portals(
        self,
        hands_by_key: Dict[str, Hand],
        per_hand: Dict[str, HandGestureState],
        dt: float,
    ) -> None:
        cfg = self.cfg.circle_trace
        finished_keys = []

        for key, portal in self.traced_portals.items():
            hand = hands_by_key.get(key)
            state = per_hand.get(key)

            if hand is not None:
                # carry the portal along with the fingertip that drew it
                fingertip = hand.point(INDEX_TIP)
                portal.set_center(fingertip, smoothing=cfg.drag_smoothing)
                # closing a fist on the carrying hand dismisses the portal,
                # same as swiping a hand shut on something you're holding
                if state is not None and state.fist:
                    portal.begin_close()

            portal.update(dt, self.cfg.spell.portal_ring_speed_deg, size_smoothing=cfg.open_close_smoothing)

            if portal.is_closed():
                finished_keys.append(key)

        for key in finished_keys:
            del self.traced_portals[key]

    # ------------------------------------------------------------------ #
    # Spells triggered by swipes
    # ------------------------------------------------------------------ #
    def _handle_swipes(self, swipes: List[SwipeEvent], hands: List[Hand], per_hand: Dict[str, HandGestureState]) -> None:
        spell_cfg = self.cfg.spell
        for swipe in swipes:
            state = per_hand.get(swipe.hand_key)
            if state is None or not state.open_palm:
                continue  # only fire spells from an open, "casting" hand

            color = self.cfg.circle.color_core
            spell = SPELL_NAMES[self.spell_index]

            if spell == "Projectile":
                vel = (swipe.direction[0] * spell_cfg.projectile_speed, swipe.direction[1] * spell_cfg.projectile_speed)
                self.projectiles.append(
                    effects.Projectile(pos=[swipe.origin[0], swipe.origin[1]], vel=vel, radius=spell_cfg.projectile_radius, color=color)
                )
                self.particles.spawn_spark(swipe.origin, self.cfg.circle.color_spark, count=8)

            elif spell == "Lightning":
                target = (swipe.origin[0] + swipe.direction[0] * 400, swipe.origin[1] + swipe.direction[1] * 400)
                self.lightning_bolts.append(effects.LightningBolt(p1=swipe.origin, p2=target))
                self.particles.spawn_spark(swipe.origin, (255, 230, 140), count=10)

            elif spell == "Beam":
                self.beams.append(
                    effects.EnergyBeam(origin=swipe.origin, direction=swipe.direction, color=color)
                )

    def _update_spells(self, dt: float) -> None:
        spell_cfg = self.cfg.spell

        for p in self.projectiles:
            p.update(dt)
        self.projectiles = [p for p in self.projectiles if p.alive(spell_cfg.projectile_lifetime)]

        self.lightning_bolts = [b for b in self.lightning_bolts if b.alive(spell_cfg.lightning_lifetime)]
        self.beams = [b for b in self.beams if b.alive(spell_cfg.beam_lifetime)]
        self.shield_throws = [s for s in self.shield_throws if s.alive(spell_cfg.beam_lifetime + 0.4)]

    # ------------------------------------------------------------------ #
    # Ambient particles: sparks around circle edges, embers, hand trails
    # ------------------------------------------------------------------ #
    def _spawn_ambient_particles(self, hands: List[Hand], per_hand: Dict[str, HandGestureState], dt: float) -> None:
        for key, circle in self.circles.items():
            if not circle.active:
                continue
            # sparks around the ring edge
            angle = np.random.uniform(0, 2 * np.pi)
            r = circle.current_radius
            edge = (circle.center[0] + math.cos(angle) * r, circle.center[1] + math.sin(angle) * r)
            self.particles.spawn_spark(edge, self.cfg.circle.color_spark, count=1)
            # slow rising embers from the core
            if np.random.random() < 0.5:
                self.particles.spawn_ember(circle.center, self.cfg.circle.color_mid, count=1)

        # energy trails following hand movement
        for i, hand in enumerate(hands):
            key = f"{hand.handedness}_{i}"
            state = per_hand.get(key)
            if state is None:
                continue
            # approximate instantaneous velocity from rotation-free center delta
            self.particles.spawn_trail(state.center, (0.0, 0.0), self.cfg.circle.color_core)

        if self.portal is not None:
            for _ in range(2):
                angle = np.random.uniform(0, 2 * np.pi)
                r = self.portal.current_radius
                edge = (self.portal.center[0] + math.cos(angle) * r, self.portal.center[1] + math.sin(angle) * r)
                self.particles.spawn_spark(edge, self.cfg.circle.color_spark, count=1)

        for portal in self.traced_portals.values():
            if portal.state not in ("opening", "open"):
                continue
            angle = np.random.uniform(0, 2 * np.pi)
            r = portal.current_radius
            edge = (portal.center[0] + math.cos(angle) * r, portal.center[1] + math.sin(angle) * r)
            self.particles.spawn_spark(edge, self.cfg.circle.color_spark, count=1)

    # ------------------------------------------------------------------ #
    # UI overlay
    # ------------------------------------------------------------------ #
    def _draw_ui(self, frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        def text(s, x, y, scale=0.6, color=(230, 230, 230), thick=1):
            cv2.putText(frame, s, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, s, (x, y), font, scale, color, thick, cv2.LINE_AA)

        text(f"FPS: {fps:5.1f}", 16, 28, 0.65, (120, 255, 120))
        text(f"Spell: {SPELL_NAMES[self.spell_index]}  (1/2/3 to switch)", 16, 54)
        active_traced = sum(1 for p in self.traced_portals.values() if p.state in ("opening", "open"))
        portal_bit = f"Portal: {'ON' if self.portal_mode else 'off'}"
        if active_traced:
            portal_bit += f" (SlingRing x{active_traced})"
        text(f"Magic: {'ON' if self.magic_mode else 'off'}   {portal_bit}   Debug: {'ON' if self.debug_mode else 'off'}", 16, 78)

        active_gestures = []
        for key, state in getattr(self, "_current_hand_states", {}).items():
            if state.open_palm:
                active_gestures.append(f"{state.handedness}:OpenPalm")
            elif state.fist:
                active_gestures.append(f"{state.handedness}:Fist")
            else:
                active_gestures.append(f"{state.handedness}:Move")
        status = " | ".join(active_gestures) if active_gestures else "No hands detected"
        text(status, 16, h - 20, 0.6, (255, 200, 120))

        if self._video_writer is not None:
            text("REC", w - 90, 30, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (w - 110, 24), 6, (0, 0, 255), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Keyboard handling
    # ------------------------------------------------------------------ #
    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        """Returns False if the app should exit."""
        if key == 27:  # ESC
            return False
        elif key in (ord("m"), ord("M")):
            self.magic_mode = not self.magic_mode
        elif key in (ord("p"), ord("P")):
            self.portal_mode = not self.portal_mode
        elif key in (ord("d"), ord("D")):
            self.debug_mode = not self.debug_mode
        elif key in (ord("f"), ord("F")):
            self._toggle_fullscreen()
        elif key in (ord("1"),):
            self.spell_index = 0
        elif key in (ord("2"),):
            self.spell_index = 1
        elif key in (ord("3"),):
            self.spell_index = 2
        elif key in (ord("c"), ord("C")):
            self._save_screenshot(frame)
        elif key in (ord("v"), ord("V")):
            self._toggle_recording(frame.shape[1], frame.shape[0])
        return True

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.cfg.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        fname = os.path.join(self.cfg.screenshot_dir, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(fname, frame)
        print(f"Saved screenshot -> {fname}")

    def _toggle_recording(self, width: int, height: int) -> None:
        if self._video_writer is None:
            fname = os.path.join(self.cfg.recording_dir, f"recording_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(fname, fourcc, 30.0, (width, height))
            print(f"Recording started -> {fname}")
        else:
            self._video_writer.release()
            self._video_writer = None
            print("Recording stopped.")

    # ------------------------------------------------------------------ #
    def _cleanup(self) -> None:
        self.cap.release()
        if self._video_writer is not None:
            self._video_writer.release()
        self.tracker.close()
        cv2.destroyAllWindows()


def main() -> None:
    app = DoctorStrangeApp(CONFIG)
    app.run()


if __name__ == "__main__":
    main()
