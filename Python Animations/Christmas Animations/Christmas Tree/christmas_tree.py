"""Colorful animated ASCII Christmas tree for the terminal.

Uses only the Python standard library and ANSI 24-bit (truecolor) escape codes.
Run with: python christmas_tree.py
Press Ctrl+C to exit cleanly.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass

RESET = "\033[0m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_SCREEN = "\033[2J\033[H"
CURSOR_HOME = "\033[H"


def rgb(r: int, g: int, b: int) -> str:
    """Return an ANSI 24-bit foreground color escape sequence."""
    return f"\033[38;2;{r};{g};{b}m"


def bold_rgb(r: int, g: int, b: int) -> str:
    """Return a bold ANSI 24-bit foreground color escape sequence."""
    return f"\033[1;38;2;{r};{g};{b}m"


def enable_windows_ansi() -> None:
    """Enable Virtual Terminal Processing on Windows consoles."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


THEMES: dict[str, dict[str, list[tuple[int, int, int]]]] = {
    "classic": {
        "foliage": [(24, 130, 52), (34, 155, 65), (18, 108, 42)],
        "ornaments": [
            (255, 59, 59),    # Ruby red
            (255, 215, 0),    # Gold
            (64, 196, 255),   # Ice blue
            (255, 105, 180),  # Magenta pink
            (255, 255, 255),  # Pearl white
        ],
        "garland": (255, 215, 90),
        "star": (255, 230, 80),
    },
    "aurora": {
        "foliage": [(18, 125, 110), (28, 160, 125), (15, 95, 90)],
        "ornaments": [
            (110, 255, 210),  # Mint
            (130, 180, 255),  # Sky
            (215, 130, 255),  # Lavender
            (255, 235, 120),  # Pale gold
            (255, 120, 190),  # Rose
        ],
        "garland": (150, 240, 255),
        "star": (220, 255, 240),
    },
    "candy": {
        "foliage": [(30, 145, 70), (45, 170, 85), (20, 115, 55)],
        "ornaments": [
            (255, 45, 85),    # Candy red
            (255, 255, 255),  # Peppermint white
            (255, 150, 200),  # Bubblegum
            (100, 235, 255),  # Frost blue
            (255, 210, 75),   # Caramel gold
        ],
        "garland": (255, 110, 150),
        "star": (255, 240, 130),
    },
}

ORNAMENT_CHARS = ["o", "O", "*", "@", "+"]
SNOW_CHARS = [".", "*", "+", "·"]


@dataclass
class Snowflake:
    x: float
    y: float
    speed: float
    drift: float
    char: str
    brightness: int


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (0..1) to RGB (0..255)."""
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def build_tree_layout(height: int, rng: random.Random) -> dict[tuple[int, int], tuple[str, int]]:
    """Precompute ornament positions and types for the tree."""
    ornaments: dict[tuple[int, int], tuple[str, int]] = {}
    for row in range(height):
        tier_offset = row % 4
        width = max(1, 1 + 2 * (row - tier_offset // 3))
        left = -width // 2
        right = width // 2
        for col in range(left, right + 1):
            if row >= 2 and abs(col) < width // 2 and rng.random() < 0.22:
                char = rng.choice(ORNAMENT_CHARS)
                phase = rng.randint(0, 100)
                ornaments[(row, col)] = (char, phase)
    return ornaments


def render_frame(
    frame_idx: int,
    tree_height: int,
    canvas_w: int,
    canvas_h: int,
    ornaments: dict[tuple[int, int], tuple[str, int]],
    snowflakes: list[Snowflake],
    theme_name: str,
    show_snow: bool,
) -> str:
    """Render a single animation frame as a string."""
    theme = THEMES[theme_name]
    center_x = canvas_w // 2
    top_y = 2

    # Grid stores (char, ansi_color)
    grid: list[list[tuple[str, str]]] = [
        [(" ", "") for _ in range(canvas_w)] for _ in range(canvas_h)
    ]

    def put(x: int, y: int, ch: str, color: str) -> None:
        if 0 <= x < canvas_w and 0 <= y < canvas_h:
            grid[y][x] = (ch, color)

    # 1. Background falling snow
    if show_snow:
        for flake in snowflakes:
            fx = int(round(flake.x)) % canvas_w
            fy = int(round(flake.y))
            if 0 <= fy < canvas_h - 1:
                b = flake.brightness
                put(fx, fy, flake.char, rgb(b, b, min(255, b + 15)))

    # 2. Star at the top (pulsing glow)
    pulse = 0.75 + 0.25 * math.sin(frame_idx * 0.25)
    sr, sg, sb = theme["star"]
    star_col = bold_rgb(int(sr * pulse), int(sg * pulse), int(sb * pulse))
    ray_col = rgb(int(sr * 0.65 * pulse), int(sg * 0.65 * pulse), int(sb * 0.55 * pulse))

    if frame_idx % 6 < 3:
        put(center_x, top_y - 1, "|", ray_col)
        put(center_x - 2, top_y, "-", ray_col)
        put(center_x + 2, top_y, "-", ray_col)
    else:
        put(center_x - 1, top_y - 1, "\\", ray_col)
        put(center_x + 1, top_y - 1, "/", ray_col)
    put(center_x - 1, top_y, "<", star_col)
    put(center_x, top_y, "*", star_col)
    put(center_x + 1, top_y, ">", star_col)

    # 3. Tree foliage, garland, and twinkling ornaments
    foliage_palette = theme["foliage"]
    ornament_palette = theme["ornaments"]
    gr, gg, gb = theme["garland"]

    for row in range(tree_height):
        y = top_y + 1 + row
        if y >= canvas_h - 4:
            break
        tier_indent = (row % 4) // 2
        half_w = max(0, row - tier_indent)
        for col in range(-half_w, half_w + 1):
            x = center_x + col
            # Diagonal swirling garland wave
            is_garland = (row + col - (frame_idx // 3)) % 5 == 0 and abs(col) < half_w
            if (row, col) in ornaments:
                ch, phase = ornaments[(row, col)]
                twinkle = 0.55 + 0.45 * math.sin(frame_idx * 0.35 + phase)
                base_r, base_g, base_b = ornament_palette[(phase + (frame_idx // 20)) % len(ornament_palette)]
                color = bold_rgb(
                    int(base_r * twinkle),
                    int(base_g * twinkle),
                    int(base_b * twinkle),
                )
                put(x, y, ch, color)
            elif is_garland:
                shimmer = 0.7 + 0.3 * math.sin(frame_idx * 0.3 + col * 0.4)
                put(x, y, "~", rgb(int(gr * shimmer), int(gg * shimmer), int(gb * shimmer)))
            else:
                if col == -half_w:
                    ch = "/"
                elif col == half_w:
                    ch = "\\"
                elif row % 4 == 3 and (col % 2 == 0):
                    ch = "^"
                else:
                    ch = "*" if (row + col) % 3 == 0 else "^"
                fr, fg, fb = foliage_palette[(row + abs(col)) % len(foliage_palette)]
                put(x, y, ch, rgb(fr, fg, fb))

    # 4. Trunk
    trunk_top = top_y + 1 + min(tree_height, max(1, canvas_h - top_y - 5))
    trunk_h = 3
    trunk_color = rgb(139, 90, 43)
    for ty in range(trunk_h):
        y = trunk_top + ty
        if y >= canvas_h - 2:
            break
        for dx in range(-2, 3):
            put(center_x + dx, y, "|" if abs(dx) == 2 else "#", trunk_color)

    # 5. Presents under the tree
    base_y = min(canvas_h - 2, trunk_top + trunk_h - 1)
    gifts = [
        (center_x - 8, "[+]", (220, 50, 60), (255, 215, 0)),
        (center_x + 6, "[*]", (65, 165, 255), (255, 255, 255)),
        (center_x - 13, "[#]", (190, 85, 245), (255, 220, 120)),
    ]
    for gx, gstr, box_col, ribbon_col in gifts:
        for i, ch in enumerate(gstr):
            c = bold_rgb(*ribbon_col) if i == 1 else rgb(*box_col)
            put(gx + i, base_y, ch, c)

    # 6. Snow ground line
    ground_y = min(canvas_h - 2, base_y + 1)
    for x in range(canvas_w):
        if grid[ground_y][x][0] == " ":
            put(x, ground_y, "_", rgb(200, 225, 255))

    # 7. Festive animated rainbow banner
    banner = "  *  M E R R Y   C H R I S T M A S  *  "
    banner_y = min(canvas_h - 1, ground_y + 1)
    start_x = max(0, center_x - len(banner) // 2)
    for i, ch in enumerate(banner):
        hue = ((i * 0.03) - (frame_idx * 0.02)) % 1.0
        r, g, b = hsv_to_rgb(hue, 0.75, 1.0)
        put(start_x + i, banner_y, ch, bold_rgb(r, g, b))

    # Flatten grid into ANSI string
    lines: list[str] = []
    for row in grid:
        out: list[str] = []
        last_color = ""
        for ch, color in row:
            if color != last_color:
                out.append(color if color else RESET)
                last_color = color
            out.append(ch)
        if last_color:
            out.append(RESET)
        lines.append("".join(out))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colorful animated ASCII Christmas tree.")
    parser.add_argument("--height", type=int, default=18, help="Tree height in rows (default: 18)")
    parser.add_argument("--fps", type=float, default=12.0, help="Frames per second (default: 12)")
    parser.add_argument(
        "--theme",
        choices=sorted(THEMES.keys()),
        default="classic",
        help="Color theme (default: classic)",
    )
    parser.add_argument("--no-snow", action="store_true", help="Disable falling snow")
    args = parser.parse_args()

    enable_windows_ansi()
    rng = random.Random()

    term_size = shutil.get_terminal_size((80, 28))
    canvas_w = max(50, min(term_size.columns - 1, 100))
    tree_height = max(8, min(args.height, term_size.lines - 8))
    canvas_h = max(tree_height + 8, min(term_size.lines - 1, tree_height + 10))

    ornaments = build_tree_layout(tree_height, rng)

    snowflakes: list[Snowflake] = []
    if not args.no_snow:
        num_flakes = max(20, canvas_w // 2)
        for _ in range(num_flakes):
            snowflakes.append(
                Snowflake(
                    x=rng.uniform(0, canvas_w),
                    y=rng.uniform(0, canvas_h - 2),
                    speed=rng.uniform(0.25, 0.65),
                    drift=rng.uniform(-0.2, 0.2),
                    char=rng.choice(SNOW_CHARS),
                    brightness=rng.randint(150, 245),
                )
            )

    frame_delay = 1.0 / max(1.0, args.fps)
    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    sys.stdout.flush()

    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            frame = render_frame(
                frame_idx=frame_idx,
                tree_height=tree_height,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                ornaments=ornaments,
                snowflakes=snowflakes,
                theme_name=args.theme,
                show_snow=not args.no_snow,
            )
            sys.stdout.write(CURSOR_HOME + frame)
            sys.stdout.flush()

            if not args.no_snow:
                for flake in snowflakes:
                    flake.y += flake.speed
                    flake.x += flake.drift + 0.1 * math.sin(frame_idx * 0.1 + flake.y)
                    if flake.y >= canvas_h - 2:
                        flake.y = 0.0
                        flake.x = rng.uniform(0, canvas_w)

            frame_idx += 1
            elapsed = time.perf_counter() - t0
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET + SHOW_CURSOR + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
