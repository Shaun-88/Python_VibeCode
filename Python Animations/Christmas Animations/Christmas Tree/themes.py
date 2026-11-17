"""Color palettes, math easing helpers, and light pattern evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

RGBf = tuple[float, float, float]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ramp(t: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if t >= end else 0.0
    return clamp01((t - start) / (end - start))


def smoothstep(t: float, start: float = 0.0, end: float = 1.0) -> float:
    u = ramp(t, start, end)
    return u * u * (3.0 - 2.0 * u)


def ease_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 1.0 - (1.0 - u) ** 3


def ease_out_back(u: float, overshoot: float = 1.6) -> float:
    u = clamp01(u)
    if u == 0.0 or u == 1.0:
        return u
    t = u - 1.0
    return 1.0 + (overshoot + 1.0) * (t**3) + overshoot * (t**2)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass(frozen=True)
class ThemePalette:
    name: str
    bg_top: RGBf
    bg_bottom: RGBf
    wall_line: RGBf
    wall_fill: RGBf
    floor_line: RGBf
    floor_fill: RGBf
    rug_outer: RGBf
    rug_inner: RGBf
    window_frame: RGBf
    window_sky_top: RGBf
    window_sky_bot: RGBf
    moon: RGBf
    trunk: RGBf
    pine_dark: RGBf
    pine_bright: RGBf
    garland: RGBf
    star: RGBf
    baubles: tuple[RGBf, ...]
    lights: tuple[RGBf, ...]
    gift_wraps: tuple[RGBf, ...]
    gift_ribbons: tuple[RGBf, ...]
    banner_a: RGBf
    banner_b: RGBf
    hearth_brick: RGBf
    hearth_mantel: RGBf
    fire_glow: RGBf


THEMES: dict[str, ThemePalette] = {
    "classic": ThemePalette(
        name="Classic",
        bg_top=(0.015, 0.020, 0.045),
        bg_bottom=(0.026, 0.032, 0.062),
        wall_line=(0.34, 0.42, 0.62),
        wall_fill=(0.14, 0.18, 0.28),
        floor_line=(0.45, 0.32, 0.22),
        floor_fill=(0.20, 0.14, 0.10),
        rug_outer=(0.68, 0.16, 0.20),
        rug_inner=(0.42, 0.10, 0.14),
        window_frame=(0.56, 0.64, 0.80),
        window_sky_top=(0.03, 0.06, 0.17),
        window_sky_bot=(0.07, 0.13, 0.30),
        moon=(1.00, 0.94, 0.64),
        trunk=(0.48, 0.29, 0.15),
        pine_dark=(0.04, 0.28, 0.11),
        pine_bright=(0.18, 0.72, 0.30),
        garland=(1.00, 0.80, 0.26),
        star=(1.00, 0.90, 0.32),
        baubles=(
            (0.95, 0.16, 0.20),
            (0.98, 0.80, 0.22),
            (0.88, 0.92, 0.98),
            (0.22, 0.62, 0.98),
        ),
        lights=(
            (1.00, 0.20, 0.24),
            (1.00, 0.84, 0.22),
            (0.24, 0.78, 1.00),
            (1.00, 0.42, 0.78),
            (1.00, 0.98, 0.88),
        ),
        gift_wraps=(
            (0.88, 0.14, 0.20),
            (0.16, 0.56, 0.90),
            (0.14, 0.66, 0.36),
            (0.68, 0.26, 0.82),
            (0.92, 0.48, 0.16),
        ),
        gift_ribbons=(
            (1.00, 0.84, 0.28),
            (0.92, 0.96, 1.00),
            (1.00, 0.84, 0.28),
            (0.92, 0.96, 1.00),
            (1.00, 0.94, 0.76),
        ),
        banner_a=(1.00, 0.30, 0.32),
        banner_b=(1.00, 0.88, 0.40),
        hearth_brick=(0.56, 0.24, 0.20),
        hearth_mantel=(0.62, 0.38, 0.22),
        fire_glow=(1.00, 0.48, 0.14),
    ),
    "candle": ThemePalette(
        name="Candlelight",
        bg_top=(0.020, 0.016, 0.030),
        bg_bottom=(0.038, 0.028, 0.040),
        wall_line=(0.46, 0.39, 0.34),
        wall_fill=(0.19, 0.15, 0.13),
        floor_line=(0.52, 0.36, 0.22),
        floor_fill=(0.24, 0.16, 0.10),
        rug_outer=(0.74, 0.42, 0.18),
        rug_inner=(0.46, 0.24, 0.12),
        window_frame=(0.62, 0.58, 0.64),
        window_sky_top=(0.03, 0.05, 0.14),
        window_sky_bot=(0.08, 0.11, 0.24),
        moon=(1.00, 0.92, 0.68),
        trunk=(0.52, 0.32, 0.16),
        pine_dark=(0.05, 0.26, 0.10),
        pine_bright=(0.20, 0.64, 0.26),
        garland=(1.00, 0.78, 0.34),
        star=(1.00, 0.94, 0.56),
        baubles=(
            (1.00, 0.82, 0.36),
            (0.92, 0.48, 0.24),
            (0.98, 0.93, 0.80),
            (0.82, 0.24, 0.22),
        ),
        lights=(
            (1.00, 0.84, 0.44),
            (1.00, 0.66, 0.24),
            (1.00, 0.96, 0.84),
            (1.00, 0.50, 0.32),
        ),
        gift_wraps=(
            (0.84, 0.26, 0.20),
            (0.88, 0.60, 0.22),
            (0.24, 0.52, 0.32),
            (0.70, 0.34, 0.24),
            (0.92, 0.78, 0.52),
        ),
        gift_ribbons=(
            (1.00, 0.90, 0.55),
            (1.00, 0.96, 0.86),
            (1.00, 0.84, 0.40),
            (1.00, 0.90, 0.55),
            (0.86, 0.28, 0.24),
        ),
        banner_a=(1.00, 0.78, 0.38),
        banner_b=(1.00, 0.96, 0.80),
        hearth_brick=(0.60, 0.28, 0.20),
        hearth_mantel=(0.66, 0.42, 0.22),
        fire_glow=(1.00, 0.56, 0.18),
    ),
    "aurora": ThemePalette(
        name="Aurora",
        bg_top=(0.012, 0.024, 0.050),
        bg_bottom=(0.022, 0.042, 0.076),
        wall_line=(0.34, 0.50, 0.68),
        wall_fill=(0.12, 0.20, 0.30),
        floor_line=(0.36, 0.44, 0.56),
        floor_fill=(0.16, 0.20, 0.28),
        rug_outer=(0.24, 0.66, 0.76),
        rug_inner=(0.12, 0.34, 0.46),
        window_frame=(0.58, 0.74, 0.88),
        window_sky_top=(0.02, 0.08, 0.18),
        window_sky_bot=(0.06, 0.18, 0.34),
        moon=(0.88, 0.97, 1.00),
        trunk=(0.42, 0.30, 0.22),
        pine_dark=(0.04, 0.30, 0.26),
        pine_bright=(0.18, 0.74, 0.58),
        garland=(0.72, 0.96, 1.00),
        star=(0.90, 1.00, 0.98),
        baubles=(
            (0.42, 0.98, 0.84),
            (0.45, 0.74, 1.00),
            (0.84, 0.56, 1.00),
            (0.95, 0.98, 1.00),
        ),
        lights=(
            (0.42, 1.00, 0.84),
            (0.45, 0.76, 1.00),
            (0.84, 0.56, 1.00),
            (1.00, 0.94, 0.64),
            (0.94, 0.98, 1.00),
        ),
        gift_wraps=(
            (0.20, 0.68, 0.86),
            (0.56, 0.38, 0.88),
            (0.18, 0.74, 0.62),
            (0.32, 0.52, 0.92),
            (0.76, 0.86, 0.96),
        ),
        gift_ribbons=(
            (0.94, 0.98, 1.00),
            (0.48, 1.00, 0.86),
            (1.00, 0.94, 0.64),
            (0.94, 0.98, 1.00),
            (0.36, 0.74, 0.98),
        ),
        banner_a=(0.44, 0.96, 0.88),
        banner_b=(0.76, 0.66, 1.00),
        hearth_brick=(0.42, 0.34, 0.44),
        hearth_mantel=(0.52, 0.44, 0.48),
        fire_glow=(1.00, 0.54, 0.22),
    ),
}

THEME_ORDER = ("classic", "candle", "aurora")
LIGHT_MODES = ("auto", "twinkle", "chase", "sparkle", "bands")
