"""High-resolution 3D ASCII projection, Z-buffer rasterizer, glyph atlas, and sharp bloom."""

from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pygame

from scene_models import (
    CAL_X1,
    CAL_X2,
    CAL_Y1,
    CAL_Y2,
    CAL_Z,
    FLOOR_Y,
    PointBatch,
    WINDOW_PANELS,
    WIN_Z,
)
from themes import ThemePalette

WIN_W = 1440
WIN_H = 900
CW = 7
CH = 12
COLS = WIN_W // CW          # 205 columns (1435px)
ROWS = WIN_H // CH          # 75 rows (900px)
SCENE_ROWS = 65             # 3D room/tree/Santa occupy rows 0..64

CAM_Z = -4.75
FOCAL = 282.0
CX = COLS * 0.50
CY = 34.0

RAMP = " .:-=+*#%@"
ATLAS_CHARS = "".join(chr(c) for c in range(32, 127))
CHAR_TO_ID = {ch: idx for idx, ch in enumerate(ATLAS_CHARS)}
RAMP_IDS = np.array([CHAR_TO_ID[ch] for ch in RAMP], dtype=np.int32)
BANNER_CHARS = np.array([CHAR_TO_ID[ch] for ch in "#%@%#"], dtype=np.int32)

FONT_5X5: dict[str, tuple[str, ...]] = {
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#"),
    "E": ("#####", "#....", "####.", "#....", "#####"),
    "R": ("####.", "#...#", "####.", "#..#.", "#...#"),
    "Y": ("#...#", ".#.#.", "..#..", "..#..", "..#.."),
    "C": (".####", "#....", "#....", "#....", ".####"),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "#####"),
    "S": (".####", "#....", ".###.", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "A": (".###.", "#...#", "#####", "#...#", "#...#"),
    " ": ("...", "...", "...", "...", "..."),
}

LIGHT_KEY = np.array([0.45, 0.55, -0.70], dtype=np.float32)
LIGHT_KEY /= np.linalg.norm(LIGHT_KEY)
LIGHT_MOON = np.array([-0.62, 0.48, -0.62], dtype=np.float32)
LIGHT_MOON /= np.linalg.norm(LIGHT_MOON)


def _find_monospace_font() -> str | None:
    candidates = (
        r"C:\Windows\Fonts\consolab.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _build_glyph_atlas() -> np.ndarray:
    """Pre-rasterize crisp Consolas glyphs into a (N_CHARS, CH, CW) float32 atlas."""
    font_path = _find_monospace_font()
    if font_path:
        font = ImageFont.truetype(font_path, 11)
    else:
        try:
            font = ImageFont.load_default(size=11)
        except TypeError:
            font = ImageFont.load_default()

    atlas = np.zeros((len(ATLAS_CHARS), CH, CW), dtype=np.float32)
    for idx, ch in enumerate(ATLAS_CHARS):
        if ch == " ":
            continue
        im = Image.new("L", (CW * 2, CH * 2), 0)
        draw = ImageDraw.Draw(im)
        draw.text((1, 0), ch, fill=255, font=font)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        # Slight contrast boost keeps fine 11pt strokes razor-sharp
        crop = np.clip(arr[0:CH, 1 : 1 + CW] * 1.15, 0.0, 1.0)
        atlas[idx, : crop.shape[0], : crop.shape[1]] = crop
    return atlas


def _build_banner_mask(text: str) -> np.ndarray:
    """Build crisp pixel-aligned 5-row ASCII-art mask for the festive banner."""
    rows = 5
    glyphs = [list(FONT_5X5[ch]) for ch in text]
    cols = sum(len(g[0]) for g in glyphs) + 2 * (len(glyphs) - 1)
    mask = np.zeros((rows, cols), dtype=bool)
    x = 0
    for g in glyphs:
        w = len(g[0])
        for r in range(rows):
            for c, px in enumerate(g[r]):
                if px == "#":
                    mask[r, x + c] = True
        x += w + 2
    return mask


class AsciiRenderer:
    """Projects 3D point clouds onto a 205x75 character grid with sharp bloom."""

    def __init__(self) -> None:
        self.atlas = _build_glyph_atlas()
        self.banner_mask = _build_banner_mask("MERRY CHRISTMAS")
        self.y_grad = np.linspace(0.0, 1.0, ROWS, dtype=np.float32)[:, None, None]

        z_w = WIN_Z - CAM_Z
        self.win_rects: list[tuple[int, int, int, int]] = []
        for x1, x2, y1, y2, _ in WINDOW_PANELS:
            c1 = max(0, int(round(CX + (x1 * FOCAL) / z_w)))
            c2 = min(COLS - 1, int(round(CX + (x2 * FOCAL) / z_w)))
            r1 = max(0, int(round(CY - (y2 * FOCAL * (CW / CH)) / z_w)))
            r2 = min(SCENE_ROWS - 1, int(round(CY - (y1 * FOCAL * (CW / CH)) / z_w)))
            self.win_rects.append((c1, c2, r1, r2))

        z_fp = 0.80 - CAM_Z
        self.fp_c1 = max(0, int(round(CX + (-1.45 * FOCAL) / z_fp)))
        self.fp_c2 = min(COLS - 1, int(round(CX + (-1.17 * FOCAL) / z_fp)))
        self.fp_r1 = max(0, int(round(CY - (-0.24 * FOCAL * (CW / CH)) / z_fp)))
        self.fp_r2 = min(SCENE_ROWS - 1, int(round(CY - (FLOOR_Y * FOCAL * (CW / CH)) / z_fp)))

        z_c = CAL_Z - CAM_Z
        self.cal_c1 = max(0, int(round(CX + (CAL_X1 * FOCAL) / z_c)))
        self.cal_c2 = min(COLS - 1, int(round(CX + (CAL_X2 * FOCAL) / z_c)))
        self.cal_r1 = max(0, int(round(CY - (CAL_Y2 * FOCAL * (CW / CH)) / z_c)))
        self.cal_r2 = min(SCENE_ROWS - 1, int(round(CY - (CAL_Y1 * FOCAL * (CW / CH)) / z_c)))
        self.cal_mid_r = max(0, int(round(CY - ((CAL_Y2 - 0.10) * FOCAL * (CW / CH)) / z_c)))

        self.char_grid = np.zeros((ROWS, COLS), dtype=np.int32)
        self.col_grid = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        self.glow_grid = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        self.winner_grid = np.full((SCENE_ROWS, COLS), -1, dtype=np.int32)
        self.out_frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self.oy = (WIN_H - ROWS * CH) // 2
        self.ox = (WIN_W - COLS * CW) // 2

        self.atlas_flat = np.ascontiguousarray(self.atlas.reshape(len(ATLAS_CHARS), CH * CW, 1))
        self._blend_buf = np.empty((ROWS * COLS, CH * CW, 3), dtype=np.float32)
        self._blend_u8 = np.empty((ROWS * COLS, CH * CW, 3), dtype=np.uint8)
        self._raster = self.out_frame[self.oy : self.oy + ROWS * CH, self.ox : self.ox + COLS * CW]
        self._raster_view = self._raster.reshape(ROWS, CH, COLS, CW, 3)
        self._s_out = pygame.image.frombuffer(self.out_frame.data, (WIN_W, WIN_H), "RGB")
        self._bloom_buf = np.empty((ROWS * CH, COLS * CW, 3), dtype=np.uint8)
        self._s_bloom = pygame.image.frombuffer(self._bloom_buf.data, (COLS * CW, ROWS * CH), "RGB")
        self._bg_grad_cache: dict[str, np.ndarray] = {}
        self._bg_cell_cache: dict[str, np.ndarray] = {}
        self._bg_raster_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def project_point(world_pt: np.ndarray) -> tuple[int, int] | None:
        z = float(world_pt[2]) - CAM_Z
        if z <= 0.25:
            return None
        col = int(round(CX + (float(world_pt[0]) * FOCAL) / z))
        row = int(round(CY - (float(world_pt[1]) * FOCAL * (CW / CH)) / z))
        if 0 <= col < COLS and 0 <= row < SCENE_ROWS:
            return col, row
        return None

    def _draw_text_row(
        self,
        row: int,
        col_start: int,
        text: str,
        rgb: tuple[float, float, float],
        glow: float = 0.0,
    ) -> None:
        if not (0 <= row < ROWS):
            return
        for i, ch in enumerate(text):
            c = col_start + i
            if 0 <= c < COLS and ch != " ":
                self.char_grid[row, c] = CHAR_TO_ID.get(ch, 0)
                self.col_grid[row, c] = rgb
                if glow > 0.0:
                    self.glow_grid[row, c] = np.array(rgb, dtype=np.float32) * glow

    def _build_bg_cells(self, palette: ThemePalette, room_reveal: float) -> np.ndarray:
        grad = self._bg_grad_cache.get(palette.name)
        if grad is None:
            bg_top = np.array(palette.bg_top, dtype=np.float32)[None, None, :]
            bg_bot = np.array(palette.bg_bottom, dtype=np.float32)[None, None, :]
            bg_col = bg_top * (np.float32(1.0) - self.y_grad) + bg_bot * self.y_grad
            grad = np.broadcast_to(bg_col, (ROWS, COLS, 3)).copy()
            grad[SCENE_ROWS:, :, :] *= np.float32(0.40)
            self._bg_grad_cache[palette.name] = grad

        bg_cells = grad.copy()
        if room_reveal > 0.35:
            win_cut_row = int(round(SCENE_ROWS * (1.0 - room_reveal)))
            wt = np.array(palette.window_sky_top, dtype=np.float32)[None, None, :]
            wb = np.array(palette.window_sky_bot, dtype=np.float32)[None, None, :]
            for c1, c2, wr1, wr2 in self.win_rects:
                r1 = max(wr1, win_cut_row)
                if wr2 > r1:
                    w_grad = np.linspace(0.0, 1.0, wr2 - r1 + 1, dtype=np.float32)[:, None, None]
                    bg_cells[r1 : wr2 + 1, c1 : c2 + 1, :] = wt * (np.float32(1.0) - w_grad) + wb * w_grad

        if room_reveal > 0.10:
            cut_row = int(round(SCENE_ROWS * (1.0 - room_reveal)))
            fr1 = max(self.fp_r1, cut_row)
            if self.fp_r2 > fr1:
                bg_cells[fr1 : self.fp_r2 + 1, self.fp_c1 : self.fp_c2 + 1, :] = np.array(
                    (0.09, 0.025, 0.010), dtype=np.float32
                )

        if room_reveal > 0.45:
            cut_row = int(round(SCENE_ROWS * (1.0 - room_reveal)))
            cr1 = max(self.cal_r1 + 1, cut_row)
            cm1 = max(self.cal_mid_r + 1, cut_row)
            if self.cal_mid_r - 1 >= cr1:
                bg_cells[cr1 : self.cal_mid_r, self.cal_c1 + 1 : self.cal_c2, :] = np.array(
                    (0.32, 0.05, 0.07), dtype=np.float32
                )
            if self.cal_r2 - 1 >= cm1:
                bg_cells[cm1 : self.cal_r2, self.cal_c1 + 1 : self.cal_c2, :] = np.array(
                    (0.11, 0.11, 0.14), dtype=np.float32
                )
        return bg_cells

    def _fill_background(self, palette: ThemePalette, room_reveal: float) -> np.ndarray:
        if room_reveal >= 1.0:
            bg_cells = self._bg_cell_cache.get(palette.name)
            bg_raster = self._bg_raster_cache.get(palette.name)
            if bg_cells is not None and bg_raster is not None:
                np.copyto(self._raster, bg_raster)
                return bg_cells
            bg_cells = self._build_bg_cells(palette, 1.0)
            bg_u8 = np.clip(bg_cells * np.float32(255.0), 0.0, 255.0).astype(np.uint8)
            s_bg = pygame.image.frombuffer(bg_u8.data, (COLS, ROWS), "RGB")
            pygame.transform.scale(s_bg, (COLS * CW, ROWS * CH), self._s_bloom)
            self._s_out.blit(self._s_bloom, (self.ox, self.oy))
            self._bg_cell_cache[palette.name] = bg_cells
            self._bg_raster_cache[palette.name] = self._raster.copy()
            return bg_cells

        bg_cells = self._build_bg_cells(palette, room_reveal)
        bg_u8 = np.clip(bg_cells * np.float32(255.0), 0.0, 255.0).astype(np.uint8)
        s_bg = pygame.image.frombuffer(bg_u8.data, (COLS, ROWS), "RGB")
        pygame.transform.scale(s_bg, (COLS * CW, ROWS * CH), self._s_bloom)
        self._s_out.blit(self._s_bloom, (self.ox, self.oy))
        return bg_cells

    def render_frame(
        self,
        batch: PointBatch,
        palette: ThemePalette,
        t: float,
        room_reveal: float,
        title_reveal: float,
        ho_bubble: tuple[np.ndarray, str, float] | None,
        show_hints: bool,
        light_mode: str,
        snow_on: bool,
    ) -> np.ndarray:
        """Render complete (WIN_H, WIN_W, 3) uint8 RGB frame ready for display.

        Returns an internal reused frame buffer (`self.out_frame`); callers retaining
        frames across multiple calls should call `.copy()`.
        """
        self.char_grid.fill(0)
        self.col_grid.fill(0.0)
        self.glow_grid.fill(0.0)

        # 1. Project & Z-buffer first, then shade ONLY the ~5,300 winning visible cells
        if len(batch.pos) > 0:
            z = batch.pos[:, 2] - CAM_Z
            valid = z > 0.3
            if np.any(valid):
                valid_idx = np.nonzero(valid)[0]
                p = batch.pos[valid_idx]
                zv = z[valid_idx]

                px = np.rint(CX + (p[:, 0] * FOCAL) / zv).astype(np.int32)
                py = np.rint(CY - (p[:, 1] * FOCAL * (CW / CH)) / zv).astype(np.int32)
                in_view = (px >= 0) & (px < COLS) & (py >= 0) & (py < SCENE_ROWS)

                if np.any(in_view):
                    px = px[in_view]
                    py = py[in_view]
                    zv = zv[in_view]
                    src_idx = valid_idx[in_view]

                    order = np.argsort(-zv)
                    self.winner_grid.fill(-1)
                    self.winner_grid[py[order], px[order]] = src_idx[order]

                    wr, wc = np.nonzero(self.winner_grid >= 0)
                    if len(wr) > 0:
                        win_src = self.winner_grid[wr, wc]
                        n = batch.nrm[win_src]
                        c = batch.col[win_src]
                        e = batch.emit[win_src]

                        d1 = np.clip(n @ LIGHT_KEY, 0.0, 1.0)
                        d2 = np.clip(n @ LIGHT_MOON, 0.0, 1.0)
                        lit = np.clip(0.24 + 0.72 * d1 + 0.22 * d2 + 0.55 * e, 0.10, 1.40)

                        ramp_f = np.clip(0.18 + 0.68 * d1 + 0.16 * d2 + 0.45 * e, 0.10, 1.0)
                        ramp_idx = np.clip(np.rint(ramp_f * (len(RAMP) - 1)).astype(np.int32), 1, len(RAMP) - 1)
                        shaded_col = np.clip(c * lit[:, None], 0.0, 1.15)

                        self.char_grid[wr, wc] = RAMP_IDS[ramp_idx]
                        self.col_grid[wr, wc] = shaded_col
                        self.glow_grid[wr, wc] = shaded_col * np.clip(e - 0.30, 0.0, 1.1)[:, None]

        # 2. Santa's playful "HO! HO! HO!" bubble in open wall space beside window
        if ho_bubble is not None:
            world_anchor, text, alpha = ho_bubble
            proj = self.project_point(world_anchor)
            if proj is not None and alpha > 0.05:
                col_s = max(1, proj[0] - len(text) // 2)
                row_s = max(1, proj[1])
                b_col = (1.0 * alpha, 0.94 * alpha, 0.62 * alpha)
                self._draw_text_row(row_s, col_s, text, b_col, glow=0.70 * alpha)

        # 2b. Small "DEC 25" wall calendar lettering on the upper-right wall
        if room_reveal > 0.45:
            cut_row = int(round(SCENE_ROWS * (1.0 - room_reveal)))
            cal_cx = (self.cal_c1 + self.cal_c2) // 2
            hdr_r = (self.cal_r1 + self.cal_mid_r) // 2
            if hdr_r >= cut_row:
                self._draw_text_row(hdr_r, cal_cx - 2, "D E C", (0.98, 0.96, 0.92), glow=0.15)
            digits_25 = (
                "###  ###",
                "  #  #  ",
                "###  ###",
                "#      #",
                "###  ###",
            )
            d_r0 = self.cal_mid_r + 1
            d_c0 = cal_cx - len(digits_25[0]) // 2 + 1
            for dr, d_line in enumerate(digits_25):
                rr = d_r0 + dr
                if rr < self.cal_r2 and rr >= cut_row:
                    self._draw_text_row(rr, d_c0, d_line, palette.banner_b, glow=0.25)

        # 3. Crisp ASCII-shaded "MERRY CHRISTMAS" banner in bottom dark apron (rows 67..71)
        if title_reveal > 0.0:
            mr, mc = self.banner_mask.shape
            c0 = (COLS - mc) // 2
            r0 = 67
            visible_cols = int(round(mc * title_reveal))
            a = np.array(palette.banner_a, dtype=np.float32)
            b = np.array(palette.banner_b, dtype=np.float32)
            for rr in range(mr):
                row_mask = self.banner_mask[rr, :visible_cols]
                if not np.any(row_mask):
                    continue
                cols_on = np.nonzero(row_mask)[0]
                wave = 0.5 + 0.5 * np.sin(t * 2.6 - cols_on * 0.14 + rr * 0.4)
                rgb_arr = a[None, :] * (1.0 - wave[:, None]) + b[None, :] * wave[:, None]
                self.char_grid[r0 + rr, c0 + cols_on] = BANNER_CHARS[(cols_on + rr) % len(BANNER_CHARS)]
                self.col_grid[r0 + rr, c0 + cols_on] = rgb_arr
                self.glow_grid[r0 + rr, c0 + cols_on] = rgb_arr * 0.38

        # 4. Subtle bottom control bar (hidden by default; toggleable with C)
        if show_hints:
            hint = (
                f"[1-3] {palette.name}   "
                f"[M] Lights: {light_mode.capitalize()}   "
                f"[S] Snow: {'On' if snow_on else 'Off'}   "
                f"[C] Hide UI   "
                f"[F] Fullscreen   "
                f"[R] Replay   "
                f"[Q] Quit"
            )
            self._draw_text_row(ROWS - 1, max(0, (COLS - len(hint)) // 2), hint, (0.44, 0.49, 0.60))

        # 5. Deep night backdrop + midnight window glass + sharp glyph compositing
        bg_cells = self._fill_background(palette, room_reveal)

        rr, cc = np.nonzero(self.char_grid)
        n_cells = len(rr)
        if n_cells > 0:
            fg = self.col_grid[rr, cc] * np.float32(255.0)
            bg = bg_cells[rr, cc] * np.float32(255.0)
            diff = (fg - bg)[:, None, :]
            buf = self._blend_buf[:n_cells]
            buf_u8 = self._blend_u8[:n_cells]
            np.matmul(self.atlas_flat[self.char_grid[rr, cc]], diff, out=buf)
            np.add(buf, bg[:, None, :], out=buf)
            np.clip(buf, 0.0, 255.0, out=buf)
            np.copyto(buf_u8, buf, casting="unsafe")
            self._raster_view[rr, :, cc, :, :] = buf_u8.reshape(n_cells, CH, CW, 3)

        # Fast cell-grid bloom blurred cleanly in PIL and SDL2-added in place
        glow_u8 = np.clip(self.glow_grid * np.float32(115.0), 0.0, 255.0).astype(np.uint8)
        pil_glow = Image.fromarray(glow_u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.3))
        glow_blur = np.ascontiguousarray(pil_glow, dtype=np.uint8)
        s_glow = pygame.image.frombuffer(glow_blur.data, (COLS, ROWS), "RGB")
        pygame.transform.smoothscale(s_glow, (COLS * CW, ROWS * CH), self._s_bloom)
        self._s_out.blit(self._s_bloom, (self.ox, self.oy), special_flags=pygame.BLEND_RGB_ADD)
        return self.out_frame
