"""Cozy 3D ASCII Christmas Room & Santa Animation.

Timeline:
1. Cozy room (floor, rug, walls, snowy midnight window) rises bottom-to-top.
2. 3D spinning Christmas tree grows bottom-to-top with garland, baubles, lights, and star.
3. 3D Santa walks in from the left, conjures 3D ribboned gifts around the tree,
   and stays in a cheerful "HO! HO! HO!" loop while "MERRY CHRISTMAS" glows below.

Run:
    pip install -r requirements.txt
    python christmas_tree.py
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import math
import os
import time
import numpy as np
from PIL import Image
import pygame

from renderer import AsciiRenderer, WIN_H, WIN_W
from scene_models import (
    FLOOR_Y,
    GiftsModel,
    PointBatch,
    RoomSceneModel,
    SantaRig,
    build_conjure_sparkles,
    concat_batches,
)
from themes import (
    LIGHT_MODES,
    THEMES,
    THEME_ORDER,
    ease_out_back,
    ease_out_cubic,
    ramp,
    smoothstep,
)


@dataclass
class StoryState:
    room_reveal: float
    trunk_grow: float
    tree_grow: float
    garland_grow: float
    ornament_grow: float
    lights_power: float
    star_grow: float
    spin_angle: float
    santa_visible: bool
    santa_x: float
    santa_z: float
    santa_yaw: float
    walk_phase: float
    stride_amp: float
    bend: float
    wave: float
    laugh_bounce: float
    sack_on_back: bool
    conjure_progress: float
    gift_progress: float
    title_reveal: float
    ho_text: str | None
    ho_alpha: float


def evaluate_story(t: float) -> StoryState:
    """Compute all animation parameters at elapsed seconds `t`."""
    # Act 1: Cozy Room builds bottom-to-top (0.0 -> 2.4s)
    room_reveal = smoothstep(t, 0.0, 2.4)

    # Act 2: Slow, lush 3D Tree growth & decoration bottom-to-top (2.4 -> 13.3s)
    trunk_grow = ease_out_cubic(ramp(t, 2.4, 3.5))
    tree_grow = smoothstep(t, 3.3, 8.1)
    garland_grow = smoothstep(t, 7.8, 11.0)
    ornament_grow = smoothstep(t, 10.6, 12.2)
    lights_power = smoothstep(t, 11.8, 12.8)
    star_grow = ease_out_back(ramp(t, 12.5, 13.3), overshoot=1.35)
    spin_angle = max(0.0, t - 3.0) * 0.30

    # Act 3: Santa walks in from off-screen left & conjures 3D gifts (13.2 -> 19.4s)
    santa_visible = t >= 13.2
    walk_u = smoothstep(t, 13.2, 16.2)
    santa_x = -2.35 + (-0.52 - (-2.35)) * walk_u
    santa_z = 0.18
    stride_amp = math.sin(walk_u * math.pi) * 0.85 if walk_u < 1.0 else 0.0
    walk_phase = max(0.0, t - 13.2) * 6.4
    sack_on_back = t < 16.3
    santa_yaw = -0.15 + 0.42 * smoothstep(t, 15.9, 16.6)

    bend = math.sin(ramp(t, 16.2, 17.8) * math.pi) * 0.72 if 16.2 <= t <= 17.8 else 0.0
    conjure_progress = ramp(t, 16.45, 19.1)
    gift_progress = ramp(t, 16.7, 19.3)

    # Act 4: MERRY CHRISTMAS reveal (19.2 -> 21.4s)
    title_reveal = smoothstep(t, 19.2, 21.4)

    # Act 5: Cheerful natural HO! HO! HO! loop (first greeting at 19.6s, then every 4.2s)
    wave = 0.0
    laugh_bounce = 0.0
    ho_text: str | None = None
    ho_alpha = 0.0

    if t >= 19.6:
        cycle_t = (t - 19.6) % 4.2
        if cycle_t < 2.4:
            wave = math.sin(ramp(cycle_t, 0.0, 2.4) * math.pi)
            laugh_bounce = abs(math.sin(cycle_t * 5.6)) * wave
            if cycle_t < 0.55:
                ho_text = "HO!"
            elif cycle_t < 1.05:
                ho_text = "HO! HO!"
            else:
                ho_text = "HO! HO! HO!"
            fade_in = ramp(cycle_t, 0.05, 0.25)
            fade_out = 1.0 - ramp(cycle_t, 2.0, 2.4)
            ho_alpha = min(fade_in, fade_out)
        else:
            laugh_bounce = 0.18 * (0.5 + 0.5 * math.sin(t * 2.2))

    return StoryState(
        room_reveal=room_reveal,
        trunk_grow=trunk_grow,
        tree_grow=tree_grow,
        garland_grow=garland_grow,
        ornament_grow=ornament_grow,
        lights_power=lights_power,
        star_grow=star_grow,
        spin_angle=spin_angle,
        santa_visible=santa_visible,
        santa_x=santa_x,
        santa_z=santa_z,
        santa_yaw=santa_yaw,
        walk_phase=walk_phase,
        stride_amp=stride_amp,
        bend=bend,
        wave=wave,
        laugh_bounce=laugh_bounce,
        sack_on_back=sack_on_back,
        conjure_progress=conjure_progress,
        gift_progress=gift_progress,
        title_reveal=title_reveal,
        ho_text=ho_text,
        ho_alpha=ho_alpha,
    )


class ChristmasDirector:
    """Combines all 3D models and renderer into a frame generator."""

    def __init__(self) -> None:
        from scene_models import TreeModel  # local import keeps startup clean

        self.room = RoomSceneModel()
        self.tree = TreeModel()
        self.gifts = GiftsModel()
        self.santa = SantaRig()
        self.renderer = AsciiRenderer()

    def render_at(
        self,
        t: float,
        theme_key: str,
        light_mode: str,
        snow_on: bool,
        show_hints: bool,
    ) -> np.ndarray:
        palette = THEMES[theme_key]
        st = evaluate_story(t)

        tree_glow = st.lights_power * (0.70 + 0.30 * st.star_grow) * (0.85 + 0.15 * math.sin(t * 2.1))
        batches: list[PointBatch] = [
            self.room.build_batch(palette, t, st.room_reveal, snow_on, tree_glow),
            self.tree.build_batch(
                palette=palette,
                t=t,
                trunk_grow=st.trunk_grow,
                tree_grow=st.tree_grow,
                garland_grow=st.garland_grow,
                ornament_grow=st.ornament_grow,
                lights_power=st.lights_power,
                star_grow=st.star_grow,
                spin_angle=st.spin_angle,
                light_mode=light_mode,
            ),
            self.gifts.build_batch(palette, st.gift_progress, ease_out_back),
        ]

        ho_bubble: tuple[np.ndarray, str, float] | None = None
        if st.santa_visible:
            santa_batch, hand_world = self.santa.build_batch(
                t=t,
                x_pos=st.santa_x,
                z_pos=st.santa_z,
                yaw=st.santa_yaw,
                walk_phase=st.walk_phase,
                stride_amp=st.stride_amp,
                bend=st.bend,
                wave=st.wave,
                laugh_bounce=st.laugh_bounce,
                sack_on_back=st.sack_on_back,
            )
            batches.append(santa_batch)
            if 0.0 < st.conjure_progress < 1.0:
                batches.append(build_conjure_sparkles(palette, hand_world, st.conjure_progress, t))
            if st.ho_text and st.ho_alpha > 0.05:
                anchor = np.array([st.santa_x + 0.32, FLOOR_Y + 0.78, st.santa_z], dtype=np.float32)
                ho_bubble = (anchor, st.ho_text, st.ho_alpha)

        scene_batch = concat_batches(batches)
        return self.renderer.render_frame(
            batch=scene_batch,
            palette=palette,
            t=t,
            room_reveal=st.room_reveal,
            title_reveal=st.title_reveal,
            ho_bubble=ho_bubble,
            show_hints=show_hints,
            light_mode=light_mode,
            snow_on=snow_on,
        )


def export_snapshots(
    director: ChristmasDirector,
    out_dir: str,
    theme_key: str,
    light_mode: str,
    snow_on: bool,
    show_hints: bool,
) -> None:
    """Render key storyboard timestamps to PNG files for inspection."""
    os.makedirs(out_dir, exist_ok=True)
    keyframes = (
        ("01_room_build", 1.7),
        ("02_tree_grow", 6.2),
        ("03_tree_lit", 13.1),
        ("04_santa_walk", 14.8),
        ("05_gifts_conjure", 17.9),
        ("06_final_loop", 21.35),
    )
    for name, ts in keyframes:
        rgb = director.render_at(ts, theme_key, light_mode, snow_on, show_hints)
        Image.fromarray(rgb, mode="RGB").save(os.path.join(out_dir, f"{name}.png"))


def _enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    except Exception:
        pass


def run_window(args: argparse.Namespace) -> None:
    director = ChristmasDirector()
    theme_idx = THEME_ORDER.index(args.theme)
    mode_idx = LIGHT_MODES.index(args.mode)
    snow_on = not args.no_snow
    show_hints = not args.hide_ui
    fullscreen = bool(args.fullscreen)

    if args.snapshot_dir:
        export_snapshots(
            director,
            args.snapshot_dir,
            THEME_ORDER[theme_idx],
            LIGHT_MODES[mode_idx],
            snow_on,
            show_hints,
        )
        return

    _enable_dpi_awareness()
    pygame.init()
    pygame.display.set_caption("Merry Christmas - 3D ASCII Room")

    def create_screen(fs: bool) -> pygame.Surface:
        flags = pygame.FULLSCREEN | pygame.SCALED if fs else pygame.RESIZABLE
        return pygame.display.set_mode((WIN_W, WIN_H), flags)

    screen = create_screen(fullscreen)
    clock = pygame.time.Clock()

    t_offset = 21.5 if args.skip_intro else 0.0
    sim_t = t_offset
    wall_start = time.monotonic()
    paused = False
    running = True

    try:
        while running:
            dt = clock.tick(args.fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_1:
                        theme_idx = 0
                    elif event.key == pygame.K_2:
                        theme_idx = 1
                    elif event.key == pygame.K_3:
                        theme_idx = 2
                    elif event.key == pygame.K_m:
                        mode_idx = (mode_idx + 1) % len(LIGHT_MODES)
                    elif event.key == pygame.K_s:
                        snow_on = not snow_on
                    elif event.key == pygame.K_h:
                        show_hints = not show_hints
                    elif event.key == pygame.K_r:
                        sim_t = 0.0
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        screen = create_screen(fullscreen)

            if not paused:
                sim_t += min(dt, 0.1)

            rgb = director.render_at(
                t=sim_t,
                theme_key=THEME_ORDER[theme_idx],
                light_mode=LIGHT_MODES[mode_idx],
                snow_on=snow_on,
                show_hints=show_hints,
            )
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            if screen.get_size() != (WIN_W, WIN_H):
                scaled = pygame.transform.smoothscale(surf, screen.get_size())
                screen.blit(scaled, (0, 0))
            else:
                screen.blit(surf, (0, 0))
            pygame.display.flip()

            if args.duration is not None and (time.monotonic() - wall_start) >= args.duration:
                running = False
    finally:
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Point-Cloud ASCII Christmas Room & Santa Animation."
    )
    parser.add_argument("--theme", choices=THEME_ORDER, default="classic", help="Starting theme")
    parser.add_argument("--mode", choices=LIGHT_MODES, default="auto", help="Starting light mode")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate (default: 30)")
    parser.add_argument("--skip-intro", action="store_true", help="Jump straight to the final loop")
    parser.add_argument("--no-snow", action="store_true", help="Start with window snow disabled")
    parser.add_argument("--hide-ui", action="store_true", help="Start in pure screensaver mode")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen")
    parser.add_argument("--duration", type=float, default=None, help="Auto-exit after N seconds")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Export storyboard PNGs and exit")
    return parser.parse_args()


def main() -> None:
    run_window(parse_args())


if __name__ == "__main__":
    main()
