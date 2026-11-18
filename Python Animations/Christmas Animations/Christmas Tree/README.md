# 3D ASCII Christmas Room & Santa Animation

A high-resolution, real-time 3D point-cloud ASCII animation built with Python, NumPy, Pillow, and Pygame-CE.

## Storyboard Sequence

1. **Cozy Room Build (Down -> Up):** Shaded 3D floor planks, woven rug, brick fireplace with crackling flames & 3 mantel stockings, chimney wreath, upper-wall fairy lights, a `"DEC 25"` wall calendar, and a 3-part bay window (big center + small side panes) with crescent moon, stars, and falling snow.
2. **3D Spinning Christmas Tree (Down -> Up):** Volumetric evergreen boughs, dark inner pine, spiraling gold garland, 3D spherical baubles, animated fairy lights, and a glowing 5-point star.
3. **Santa Arrives & Conjures Gifts:** 3D Santa walks in from the left carrying his gift sack, conjures 5 ribboned 3D gift boxes out of the rug beside the tree with golden sparkles, and enters a cheerful `"HO! HO! HO!"` loop.
4. **Banner & Endless Loop:** Shaded ASCII `"MERRY CHRISTMAS"` reveals below while the tree spins, lights twinkle, snow drifts outside, and Santa waves continuously.

## Setup & Run

```bash
pip install -r requirements.txt
python christmas_tree.py
```

Export directly to MP4 (requires `ffmpeg` on `PATH`):

```bash
python christmas_tree.py --record christmas_room.mp4 --hide-ui
```

### Live Controls

| Key | Action |
| :--- | :--- |
| `1`, `2`, `3` | Switch theme (`Classic`, `Candlelight`, `Aurora`) |
| `M` | Cycle light animation mode (`Auto`, `Twinkle`, `Chase`, `Sparkle`, `Bands`) |
| `S` | Toggle snowfall outside the window |
| `H` | Hide/show bottom control bar (pure screensaver view) |
| `F` | Toggle fullscreen |
| `Space` | Pause / resume |
| `R` | Replay full storyboard from the beginning |
| `Q` / `Esc` | Quit cleanly |

### Project Files

- `christmas_tree.py` — Storyboard timeline, DPI-aware `1440×900` `pygame-ce` window loop, and keyboard controls
- `scene_models.py` — High-density 3D point-cloud builders (~50k-point tree, room, window, gifts, and articulated Santa)
- `renderer.py` — `205×75` 3D Z-buffer rasterizer, ASCII luminance shading, crisp `Consolas` glyph atlas, and tight bloom
- `themes.py` — Color palettes and easing functions
- `requirements.txt` — Package dependencies (`numpy`, `pillow`, `pygame-ce`)
