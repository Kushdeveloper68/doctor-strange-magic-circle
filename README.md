<div align="center">

# 🔮 Doctor Strange Magic Circle

**Real-time webcam hand tracking that conjures Doctor-Strange-style glowing magic circles, portals, and spells around your bare hands.**

Pure Python. No Unity, no game engine, no green-screen tricks — just OpenCV, MediaPipe, and NumPy turning your webcam into a sanctum sanctorum.

[![Python](https://img.shields.io/badge/Python-3.9%E2%80%933.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with AI](https://img.shields.io/badge/Built%20with-Claude%20AI-8A63D2)](https://claude.ai)

<br>

<!--
  📸 Replace this with an actual GIF/screenshot once you record one!
  Suggested: record a 5-10s clip of the circle + portal + a swipe spell,
  convert to GIF, drop it in a /docs or /media folder, and update the path.
-->
<img src="docs/demo.gif" alt="Doctor Strange Magic Circle demo" width="720">

*(Add your own demo GIF here — see [Recording a demo](#-recording-a-demo) below)*

</div>

---

## 📖 Table of Contents

- [What is this?](#-what-is-this)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Was Built](#-how-it-was-built)
- [Use Cases](#-use-cases)
- [Demo / Screenshots](#-demo--screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [Controls & Gestures](#-controls--gestures)
- [Project Structure](#-project-structure)
- [Configuration / Tuning](#-configuration--tuning)
- [How the Glow Effect Works](#-how-the-glow-effect-works)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ What is this?

This project uses **MediaPipe's hand-landmark model** to track your hands through a regular webcam, then renders **rotating, glowing magic circles** (à la Doctor Strange) anchored to your palms — reacting live to gestures like opening your palm, pinching, rotating your wrist, or swiping to fire spells.

No markers, no gloves, no green screen. Just your hands and a camera.

---

## 🚀 Features

- **Real-time hand tracking** — up to 2 hands, all 21 MediaPipe landmarks, jitter-smoothed, 30+ FPS on normal hardware.
- **Rotating magic circle** — 3 independently-rotating rings, 12 procedurally-generated rune glyphs (no image assets needed), radial spokes, and a rotating hexagon/triangle core sigil.
- **Gesture-driven interaction**

  | Gesture | Effect |
  |---|---|
  | ✋ Open palm | Spawns/activates a magic circle on that hand |
  | ✊ Closed fist | Deactivates that hand's circle |
  | 🤏 Pinch (thumb + index) | Resizes the active circle |
  | 🔄 Rotate wrist | Manually rotates the circle |
  | 🙌 Two open hands | Spawns a swirling portal between them |
  | ↔️ Hands apart / together | Grows / shrinks the portal |
  | 👋 Swipe | Fires the currently selected spell |

- **Particle system** — fire sparks off the ring edges, floating embers, hand-motion energy trails, fade-over-life.
- **Cinematic visual effects** — additive-blend glow layer + real Gaussian bloom pass, motion-blur via frame persistence, orange/red/gold color grading.
- **Multiple spells** — Projectile, Lightning Bolt, Energy Beam (switchable with `1`/`2`/`3`), plus a bonus Shield-Throw animation.
- **Live UI** — FPS counter, active-gesture status line, on-screen mode indicators.
- **Bonus tools** — one-key screenshot capture and MP4 video recording, built right in.

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9 – 3.12 |
| Hand tracking | [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (legacy `solutions` API) |
| Video I/O & rendering | [OpenCV](https://opencv.org/) |
| Math / particle physics | [NumPy](https://numpy.org/) |
| Effects pipeline | Custom additive-blend + Gaussian bloom, written from scratch in `effects.py` |

No GPU, no shaders, no game engine required — it runs entirely on CPU.

---

## 🤖 How It Was Built

This project was designed and coded collaboratively with **Claude (Anthropic's AI assistant)** — from architecture to the final bug fix.

**The process, roughly:**
1. Started from a feature/architecture spec (hand tracking → gestures → magic circle → particles → spells → UI).
2. Claude generated the full module layout (`hand_tracker.py`, `gestures.py`, `magic_circle.py`, `particles.py`, `effects.py`, `main.py`) with a shared `config.py` so every visual/gesture parameter is tunable in one place instead of scattered magic numbers.
3. Every module was **actually executed and smoke-tested** in a sandbox (imports, a full mocked-camera integration run, bloom/particle output shapes) — not just generated and hoped for.
4. Along the way, a real breaking change was caught: newer `mediapipe` releases removed the legacy `mediapipe.solutions` API this project depends on, so the dependency was pinned to a known-working range (`>=0.10.9,<=0.10.21`) after verifying it against the actual installed package.
5. Iterated on gesture math (orientation-agnostic finger-extension detection, angle-wrap-safe wrist rotation, swipe-consistency checks) so it holds up regardless of how the hand is oriented to the camera.

If you're curious how any specific effect works (the bloom, the rune generation, the portal swirl), the code is heavily commented — read [How the Glow Effect Works](#-how-the-glow-effect-works) below for the short version.

---

## 💡 Use Cases

- **Portfolio / resume project** — a strong "I can do real-time CV + creative coding" showcase piece.
- **Live demos & meetups** — a fun, visually striking crowd-pleaser for tech talks, hackathon booths, or college fests.
- **Learning resource** — a clean, well-commented reference for MediaPipe hand tracking, gesture recognition math, and building a bloom/glow pipeline without a game engine.
- **Streaming / content creation** — run it as a fun webcam overlay for streams or short-form video content.
- **Base for AR/VR experiments** — the gesture recognizer and particle system are decoupled from rendering, so they can be repurposed for other hand-driven interactive projects (games, art installations, kiosks).
- **Halloween / cosplay content** 🎃 — because obviously.

---

## 📸 Demo / Screenshots

> Add screenshots or a GIF here once you've recorded one. Suggested layout:

```markdown
<p align="center">
  <img src="docs/circle.png" width="32%">
  <img src="docs/portal.png" width="32%">
  <img src="docs/spell.png" width="32%">
</p>
```

### Recording a demo
Press **`V`** while the app is running to start/stop an MP4 recording (saved to `captures/`), or **`C`** to grab a screenshot. Trim a good clip and convert to GIF, e.g.:

```bash
ffmpeg -i captures/recording_XXXX.mp4 -vf "fps=15,scale=720:-1" -loop 0 docs/demo.gif
```

---

## 📦 Installation

### Prerequisites
- Python **3.9 – 3.12** (MediaPipe does not yet reliably support 3.13+)
- A working webcam

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Kushdeveloper68/doctor-strange-magic-circle.git
cd doctor-strange-magic-circle

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Important:** `requirements.txt` pins `mediapipe<=0.10.21`. Newer MediaPipe releases removed the legacy `mediapipe.solutions` API this project relies on — don't blindly upgrade it.

---

## ▶️ Usage

```bash
python main.py
```

A window titled **"Doctor Strange Magic Circle"** opens showing your mirrored webcam feed. Hold up an open palm and watch the circle spawn.

---

## 🎮 Controls & Gestures

### Keyboard

| Key | Action |
|---|---|
| `M` | Toggle magic-circle rendering |
| `P` | Toggle portal mode |
| `D` | Toggle debug landmark overlay |
| `F` | Toggle fullscreen |
| `1` / `2` / `3` | Select spell: Projectile / Lightning / Beam |
| `C` | Save a screenshot to `captures/` |
| `V` | Start / stop MP4 recording to `captures/` |
| `Esc` | Exit |

### Hand gestures

- **Open palm** → glowing circle spawns/attaches to that hand
- **Fist** → that hand's circle deactivates
- **Pinch thumb + index** (while palm open) → circle shrinks/grows
- **Twist wrist** → circle rotates to match
- **Both hands open** → portal appears between them; move hands apart/together to resize
- **Swipe** an open hand → fires the currently selected spell from that position

---

## 🗂 Project Structure

```
doctor_strange/
│
├── main.py            # app entry point, capture loop, UI, key handling
├── hand_tracker.py     # MediaPipe wrapper + landmark smoothing
├── gestures.py          # open-palm / fist / pinch / rotation / swipe detection
├── magic_circle.py       # rotating rings, procedural runes, radial spokes
├── particles.py           # spark / ember / trail particle system
├── effects.py              # glow/bloom primitives, portal, projectile, lightning, beam
├── config.py                # every tunable parameter lives here
├── assets/
│   ├── runes/                 # reserved — runes are procedural, no files required
│   ├── textures/
│   └── sounds/
└── requirements.txt
```

All rune/sigil art is generated procedurally in `magic_circle.py` — zero external image/font assets needed to run.

---

## ⚙️ Configuration / Tuning

Every visual and gesture parameter — ring speeds, colors, particle counts, gesture sensitivity, bloom intensity — lives in **`config.py`** as typed dataclasses. That's the first (and usually only) file you need to edit to re-theme colors or adjust feel.

Useful knobs if performance is low:
```python
TrackingConfig.model_complexity = 0   # lighter MediaPipe model
CameraConfig.width, CameraConfig.height = 960, 540   # lower resolution
BloomConfig.passes = 1                # cheaper bloom
ParticleConfig.max_particles = 200    # fewer particles
```

---

## 🌈 How the Glow Effect Works

No GPU shaders — it's a classic bloom trick done entirely with OpenCV so it runs on any laptop CPU:

1. All circle/spell/particle geometry is drawn **additively** onto a black float32 "glow layer" (not directly onto the camera frame).
2. Bright pixels are thresholded out, Gaussian-blurred at increasing kernel sizes, and added back onto the sharp layer (`effects.apply_bloom`).
3. A one-frame **persistence buffer** blends each frame's glow layer with a decayed copy of the previous frame — free trailing "energy trail" motion blur.
4. The bloomed glow layer is **added** (not alpha-composited) onto the live camera frame, so overlapping glows blow out toward white-hot, just like fire/energy VFX in the films.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open webcam` | Another app may be using the camera, or try `CameraConfig.device_index = 1` in `config.py` |
| Low FPS | Lower resolution / `model_complexity = 0` / fewer particles (see [Tuning](#-configuration--tuning)) |
| `mediapipe` import errors / `AttributeError: module 'mediapipe' has no attribute 'solutions'` | You installed a mediapipe version newer than `0.10.21` — reinstall with `pip install "mediapipe<=0.10.21"` |
| Gestures feel laggy or jittery | Adjust `TrackingConfig.smoothing_alpha` in `config.py`; also check lighting on your hands |
| macOS: camera permission prompt / black window | Grant camera access to your terminal/VS Code in System Settings → Privacy & Security → Camera |

---

## 🗺 Roadmap

- [ ] Sound effects on spell cast / circle spawn (`assets/sounds/` is already scaffolded)
- [ ] Custom rune texture support (`assets/runes/`)
- [ ] True OpenGL/GLSL bloom pipeline as an optional high-quality mode
- [ ] Multiplayer / dual-webcam portal linking
- [ ] Web build via `mediapipe.js` + WebGL

Contributions toward any of these are very welcome.

---

## 🤝 Contributing

Pull requests are welcome! For larger changes, please open an issue first to discuss what you'd like to change.

```bash
# Fork, then:
git checkout -b feature/your-feature-name
# make your changes
git commit -m "Add: your feature"
git push origin feature/your-feature-name
# open a PR
```

---

## 📄 License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) by Google, for the hand-tracking model
- [OpenCV](https://opencv.org/) for making CPU-only real-time video effects possible
- Built with the assistance of **[Claude](https://claude.ai)** by Anthropic
- Inspired by the magic circle VFX from Marvel's *Doctor Strange* 🔶

<div align="center">

**If you build something cool with this, tag it — I'd love to see it.**

⭐ Star this repo if it helped you or made you smile.

</div>
