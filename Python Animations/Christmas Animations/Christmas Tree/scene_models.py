"""High-density 3D point-cloud models for the room, window, tree, gifts, and Santa."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from themes import ThemePalette, clamp01, rot_y, rot_z

FLOOR_Y = -0.78
BACK_WALL_Z = 0.85
TREE_X = 0.74
TREE_Z = 0.14

WIN_Z = BACK_WALL_Z - 0.015
CAL_X1, CAL_X2 = 1.24, 1.58
CAL_Y1, CAL_Y2 = 0.25, 0.59
CAL_Z = BACK_WALL_Z - 0.015
# (x1, x2, y1, y2, has_vertical_mullion) -> Big center window flanked by 2 small side windows
WINDOW_PANELS = (
    (-0.90, -0.68, 0.14, 0.70, False),
    (-0.63, -0.15, 0.08, 0.76, True),
    (-0.10, 0.12, 0.14, 0.70, False),
)
WIN_X1, WIN_X2 = -0.90, 0.12
WIN_Y1, WIN_Y2 = 0.08, 0.76
WIN_MX = -0.39
WIN_MY = 0.42

FIRE_X = -1.31

GIFT_SPECS = (
    # (world_x, world_z, w, h, d, yaw, wrap_idx, delay)
    (TREE_X - 0.50, TREE_Z + 0.08, 0.23, 0.24, 0.22, 0.30, 0, 0.00),
    (TREE_X - 0.25, TREE_Z - 0.28, 0.21, 0.18, 0.20, -0.35, 1, 0.18),
    (TREE_X + 0.44, TREE_Z - 0.18, 0.26, 0.25, 0.23, -0.22, 2, 0.36),
    (TREE_X + 0.56, TREE_Z + 0.18, 0.20, 0.18, 0.19, 0.42, 3, 0.54),
    (TREE_X + 0.12, TREE_Z - 0.36, 0.20, 0.16, 0.18, 0.14, 4, 0.72),
)


def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)],
        axis=1,
    ).astype(np.float32)


def _blob(
    rng: np.random.Generator,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    n: int,
    shell_bias: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-8
    dir_v = u / norms
    r_mag = (rng.random(n, dtype=np.float32) ** shell_bias)[:, None]
    rad = np.array(radii, dtype=np.float32)[None, :]
    pts = np.array(center, dtype=np.float32)[None, :] + dir_v * rad * r_mag
    nrm = dir_v / (rad + 1e-6)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
    return pts, nrm


def _tube(
    rng: np.random.Generator,
    a: tuple[float, float, float] | np.ndarray,
    b: tuple[float, float, float] | np.ndarray,
    radius: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(a, dtype=np.float32)
    pb = np.asarray(b, dtype=np.float32)
    ab = pb - pa
    length = float(np.linalg.norm(ab)) + 1e-8
    axis = ab / length
    ref = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(axis[0]) < 0.85
        else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1) + 1e-8
    e2 = np.cross(axis, e1)

    t = rng.random(n, dtype=np.float32)[:, None]
    ang = rng.uniform(0.0, math.tau, n).astype(np.float32)[:, None]
    radial = np.cos(ang) * e1 + np.sin(ang) * e2
    r_frac = (0.35 + 0.65 * np.sqrt(rng.random(n, dtype=np.float32)))[:, None]
    pts = pa[None, :] + t * ab[None, :] + radius * r_frac * radial
    return pts, radial.astype(np.float32)


def _line_pts(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    density: float = 135.0,
    normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> tuple[np.ndarray, np.ndarray]:
    pa = np.array(a, dtype=np.float32)
    pb = np.array(b, dtype=np.float32)
    dist = float(np.linalg.norm(pb - pa))
    n = max(6, int(dist * density))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    pts = pa[None, :] + t * (pb - pa)[None, :]
    nrm = np.tile(np.array(normal, dtype=np.float32), (n, 1))
    return pts, nrm


@dataclass
class PointBatch:
    pos: np.ndarray
    nrm: np.ndarray
    col: np.ndarray
    emit: np.ndarray


def empty_batch() -> PointBatch:
    z3 = np.zeros((0, 3), dtype=np.float32)
    z1 = np.zeros((0,), dtype=np.float32)
    return PointBatch(z3, z3, z3, z1)


def concat_batches(batches: list[PointBatch]) -> PointBatch:
    non_empty = [b for b in batches if len(b.pos) > 0]
    if not non_empty:
        return empty_batch()
    return PointBatch(
        pos=np.concatenate([b.pos for b in non_empty], axis=0),
        nrm=np.concatenate([b.nrm for b in non_empty], axis=0),
        col=np.concatenate([b.col for b in non_empty], axis=0),
        emit=np.concatenate([b.emit for b in non_empty], axis=0),
    )


class RoomSceneModel:
    """Builds room walls, floor, fireplace, 3-part snowy window, wall lights, and rug."""

    def __init__(self, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)

        wall_segs = [
            ((-1.72, FLOOR_Y, BACK_WALL_Z), (1.72, FLOOR_Y, BACK_WALL_Z), (0, 0.3, -1)),
            ((-1.72, 0.88, BACK_WALL_Z), (1.72, 0.88, BACK_WALL_Z), (0, -0.3, -1)),
            ((-1.72, FLOOR_Y, BACK_WALL_Z), (-1.72, 0.88, BACK_WALL_Z), (0.6, 0, -0.8)),
            ((1.72, FLOOR_Y, BACK_WALL_Z), (1.72, 0.88, BACK_WALL_Z), (-0.6, 0, -0.8)),
            ((-1.72, FLOOR_Y, BACK_WALL_Z), (-2.35, FLOOR_Y, -0.45), (0.8, 0.4, -0.4)),
            ((-1.72, 0.88, BACK_WALL_Z), (-2.35, 1.02, -0.45), (0.8, -0.4, -0.4)),
            ((1.72, FLOOR_Y, BACK_WALL_Z), (2.35, FLOOR_Y, -0.45), (-0.8, 0.4, -0.4)),
            ((1.72, 0.88, BACK_WALL_Z), (2.35, 1.02, -0.45), (-0.8, -0.4, -0.4)),
        ]
        w_pos, w_nrm = zip(*[_line_pts(a, b, density=145.0, normal=n) for a, b, n in wall_segs])
        self.wall_line_pos = np.concatenate(w_pos, axis=0)
        self.wall_line_nrm = np.concatenate(w_nrm, axis=0)

        # Quiet subtle wall texture points (skipping window & fireplace openings)
        n_wall = 1050
        wx = rng.uniform(-1.70, 1.70, n_wall).astype(np.float32)
        wy = rng.uniform(FLOOR_Y + 0.03, 0.86, n_wall).astype(np.float32)
        in_win = (wx >= WIN_X1 - 0.02) & (wx <= WIN_X2 + 0.02) & (wy >= WIN_Y1 - 0.04) & (wy <= WIN_Y2 + 0.02)
        in_fp = (wx >= -1.58) & (wx <= -1.04) & (wy <= 0.88)
        keep_w = (~in_win) & (~in_fp)
        wx, wy = wx[keep_w], wy[keep_w]
        self.wall_fill_pos = np.stack([wx, wy, np.full_like(wx, BACK_WALL_Z + 0.01)], axis=1)
        self.wall_fill_nrm = np.tile(np.array([0.0, -0.5, 0.85], dtype=np.float32), (len(wx), 1))

        floor_segs = [
            ((-2.10, FLOOR_Y, -0.35), (2.10, FLOOR_Y, -0.35), (0, 1, 0)),
        ]
        for bx in np.linspace(-1.55, 1.55, 11):
            floor_segs.append(
                ((float(bx), FLOOR_Y, BACK_WALL_Z), (float(bx) * 1.32, FLOOR_Y, -0.55), (0, 1, 0))
            )
        f_pos, f_nrm = zip(*[_line_pts(a, b, density=105.0, normal=n) for a, b, n in floor_segs])
        self.floor_line_pos = np.concatenate(f_pos, axis=0)
        self.floor_line_nrm = np.concatenate(f_nrm, axis=0)

        n_floor = 900
        fx = rng.uniform(-2.10, 2.10, n_floor).astype(np.float32)
        fz = rng.uniform(-0.55, BACK_WALL_Z - 0.01, n_floor).astype(np.float32)
        self.floor_fill_pos = np.stack([fx, np.full_like(fx, FLOOR_Y - 0.002), fz], axis=1)
        self.floor_fill_nrm = np.tile(np.array([0.0, -0.3, 0.9], dtype=np.float32), (n_floor, 1))

        # Cozy elliptical rug under tree & gifts
        n_rug = 4200
        ang = rng.uniform(0.0, math.tau, n_rug).astype(np.float32)
        rad = np.sqrt(rng.random(n_rug, dtype=np.float32))
        rx, rz = 0.80, 0.52
        rug_x = TREE_X + rx * rad * np.cos(ang)
        rug_z = TREE_Z + rz * rad * np.sin(ang)
        self.rug_pos = np.stack([rug_x, np.full_like(rug_x, FLOOR_Y + 0.003), rug_z], axis=1)
        self.rug_nrm = np.tile(np.array([0.0, 0.35, 0.9], dtype=np.float32), (n_rug, 1))
        self.rug_is_border = rad > 0.88

        # Fireplace: stone hearth slab, brick surround, chimney breast, mantel & firebox
        fp_z = BACK_WALL_Z - 0.05
        brick_segs = [
            # Chimney vertical edges & brick courses above mantel (y = -0.07 .. 0.88)
            ((-1.48, -0.07, fp_z), (-1.48, 0.88, fp_z), (0, 0, -1)),
            ((-1.14, -0.07, fp_z), (-1.14, 0.88, fp_z), (0, 0, -1)),
            # Lower surround outer posts
            ((-1.56, FLOOR_Y, fp_z), (-1.56, -0.10, fp_z), (0, 0, -1)),
            ((-1.06, FLOOR_Y, fp_z), (-1.06, -0.10, fp_z), (0, 0, -1)),
            # Firebox opening frame
            ((-1.45, FLOOR_Y + 0.02, fp_z - 0.01), (-1.45, -0.24, fp_z - 0.01), (0, 0, -1)),
            ((-1.17, FLOOR_Y + 0.02, fp_z - 0.01), (-1.17, -0.24, fp_z - 0.01), (0, 0, -1)),
            ((-1.45, -0.24, fp_z - 0.01), (-1.17, -0.24, fp_z - 0.01), (0, -0.4, -0.9)),
        ]
        # Subtle horizontal brick lines on chimney & surround
        for by in (0.12, 0.32, 0.60, 0.74):
            brick_segs.append(((-1.47, by, fp_z), (-1.15, by, fp_z), (0, 0.2, -1)))
        for by in (-0.58, -0.38):
            brick_segs.append(((-1.55, by, fp_z), (-1.46, by, fp_z), (0, 0.2, -1)))
            brick_segs.append(((-1.16, by, fp_z), (-1.07, by, fp_z), (0, 0.2, -1)))
        bp, bn = zip(*[_line_pts(a, b, density=135.0, normal=n) for a, b, n in brick_segs])
        self.fp_brick_pos = np.concatenate(bp, axis=0)
        self.fp_brick_nrm = np.concatenate(bn, axis=0)

        # Brick face texture dots on surround & chimney
        n_fb = 420
        bx = rng.uniform(-1.55, -1.07, n_fb).astype(np.float32)
        by = rng.uniform(FLOOR_Y + 0.03, 0.86, n_fb).astype(np.float32)
        in_open = (bx >= -1.46) & (bx <= -1.16) & (by <= -0.23)
        out_chim = (by > -0.07) & ((bx < -1.48) | (bx > -1.14))
        keep_fb = (~in_open) & (~out_chim)
        self.fp_brick_fill_pos = np.stack([bx[keep_fb], by[keep_fb], np.full(int(keep_fb.sum()), fp_z + 0.005, dtype=np.float32)], axis=1)
        self.fp_brick_fill_nrm = np.tile(np.array([0.0, -0.2, 0.9], dtype=np.float32), (int(keep_fb.sum()), 1))

        # Wooden mantel shelf + stone floor hearth slab
        mantel_segs = [
            ((-1.60, -0.09, fp_z - 0.05), (-1.02, -0.09, fp_z - 0.05), (0, 0.6, -0.8)),
            ((-1.60, -0.11, fp_z - 0.05), (-1.02, -0.11, fp_z - 0.05), (0, -0.3, -0.9)),
            ((-1.62, FLOOR_Y + 0.015, fp_z - 0.14), (-1.00, FLOOR_Y + 0.015, fp_z - 0.14), (0, 0.8, -0.6)),
            ((-1.62, FLOOR_Y + 0.015, fp_z - 0.14), (-1.62, FLOOR_Y + 0.01, fp_z + 0.04), (0.6, 0.5, -0.5)),
            ((-1.00, FLOOR_Y + 0.015, fp_z - 0.14), (-1.00, FLOOR_Y + 0.01, fp_z + 0.04), (-0.6, 0.5, -0.5)),
        ]
        mp, mn = zip(*[_line_pts(a, b, density=145.0, normal=n) for a, b, n in mantel_segs])
        self.fp_mantel_pos = np.concatenate(mp, axis=0)
        self.fp_mantel_nrm = np.concatenate(mn, axis=0)

        # Charred firewood logs in hearth
        log1_p, log1_n = _tube(rng, (-1.40, FLOOR_Y + 0.05, fp_z + 0.02), (-1.22, FLOOR_Y + 0.07, fp_z), 0.028, 180)
        log2_p, log2_n = _tube(rng, (-1.38, FLOOR_Y + 0.08, fp_z), (-1.24, FLOOR_Y + 0.04, fp_z + 0.03), 0.025, 160)
        self.fp_log_pos = np.concatenate([log1_p, log2_p], axis=0)
        self.fp_log_nrm = np.concatenate([log1_n, log2_n], axis=0)

        # Animated 3D flame & ember particle seeds inside firebox
        n_flames = 360
        self.fire_u = rng.random(n_flames, dtype=np.float32)          # height param [0..1]
        self.fire_x0 = rng.uniform(-0.10, 0.10, n_flames).astype(np.float32)
        self.fire_phase = rng.uniform(0.0, math.tau, n_flames).astype(np.float32)
        self.fire_freq = rng.uniform(4.0, 9.5, n_flames).astype(np.float32)

        n_sparks = 18
        self.fp_spark_x0 = rng.uniform(-0.08, 0.08, n_sparks).astype(np.float32)
        self.fp_spark_t0 = rng.random(n_sparks, dtype=np.float32)
        self.fp_spark_spd = rng.uniform(0.45, 0.85, n_sparks).astype(np.float32)

        # Chimney wreath (green pine ring + red ribbon bow)
        w_ang = rng.uniform(0.0, math.tau, 420).astype(np.float32)
        w_rad = rng.uniform(0.068, 0.122, 420).astype(np.float32)
        wr_p = np.stack(
            [FIRE_X + w_rad * np.cos(w_ang), 0.43 + w_rad * np.sin(w_ang), np.full(420, fp_z - 0.02, dtype=np.float32)],
            axis=1,
        )
        wr_n = np.stack([np.cos(w_ang) * 0.4, np.sin(w_ang) * 0.4, np.full(420, -0.85, dtype=np.float32)], axis=1)
        bow_p, bow_n = _blob(rng, (FIRE_X, 0.31, fp_z - 0.03), (0.042, 0.024, 0.015), 85, 0.6)
        self.wreath_pos = np.concatenate([wr_p, bow_p], axis=0).astype(np.float32)
        self.wreath_nrm = np.concatenate([wr_n, bow_n], axis=0).astype(np.float32)
        self.wreath_is_bow = np.concatenate([np.zeros(420, dtype=bool), np.ones(len(bow_p), dtype=bool)])

        # 3 neat Christmas stockings hung evenly along the fireplace mantel
        st_pos, st_nrm, st_col = [], [], []
        stocking_specs = (
            (-1.48, (0.90, 0.16, 0.20)),
            (-1.31, (0.16, 0.68, 0.34)),
            (-1.14, (0.96, 0.72, 0.20)),
        )
        for sx, main_rgb in stocking_specs:
            cuff_p, cuff_n = _blob(rng, (sx, -0.130, fp_z - 0.05), (0.028, 0.016, 0.015), 75, 0.6)
            leg_p, leg_n = _tube(rng, (sx, -0.145, fp_z - 0.05), (sx, -0.215, fp_z - 0.05), 0.020, 95)
            toe_p, toe_n = _blob(rng, (sx + 0.022, -0.220, fp_z - 0.05), (0.029, 0.018, 0.016), 75, 0.6)
            st_pos.extend([cuff_p, leg_p, toe_p])
            st_nrm.extend([cuff_n, leg_n, toe_n])
            st_col.extend([
                np.tile(np.array((0.95, 0.96, 0.99), dtype=np.float32), (len(cuff_p), 1)),
                np.tile(np.array(main_rgb, dtype=np.float32), (len(leg_p), 1)),
                np.tile(np.array(main_rgb, dtype=np.float32), (len(toe_p), 1)),
            ])
        self.stocking_pos = np.concatenate(st_pos, axis=0).astype(np.float32)
        self.stocking_nrm = np.concatenate(st_nrm, axis=0).astype(np.float32)
        self.stocking_col = np.concatenate(st_col, axis=0).astype(np.float32)

        # Small "DEC 25" wall calendar on the upper-right wall beside the tree
        cal_mid_x = 0.5 * (CAL_X1 + CAL_X2)
        cal_segs = [
            ((CAL_X1, CAL_Y1, CAL_Z), (CAL_X2, CAL_Y1, CAL_Z), (0, -0.2, -1)),
            ((CAL_X1, CAL_Y2, CAL_Z), (CAL_X2, CAL_Y2, CAL_Z), (0, 0.2, -1)),
            ((CAL_X1, CAL_Y1, CAL_Z), (CAL_X1, CAL_Y2, CAL_Z), (0.2, 0, -1)),
            ((CAL_X2, CAL_Y1, CAL_Z), (CAL_X2, CAL_Y2, CAL_Z), (-0.2, 0, -1)),
            ((CAL_X1, CAL_Y2 - 0.10, CAL_Z), (CAL_X2, CAL_Y2 - 0.10, CAL_Z), (0, 0, -1)),
            ((cal_mid_x - 0.08, CAL_Y2, CAL_Z), (cal_mid_x, CAL_Y2 + 0.045, CAL_Z), (0, 0.2, -1)),
            ((cal_mid_x + 0.08, CAL_Y2, CAL_Z), (cal_mid_x, CAL_Y2 + 0.045, CAL_Z), (0, 0.2, -1)),
        ]
        cp, cn = zip(*[_line_pts(a, b, density=140.0, normal=n) for a, b, n in cal_segs])
        self.cal_frame_pos = np.concatenate(cp, axis=0)
        self.cal_frame_nrm = np.concatenate(cn, axis=0)

        # Restrained scalloped fairy string lights along upper wall trim
        swags = ((-0.98, -0.52), (-0.52, -0.06), (-0.06, 0.40), (1.10, 1.62))
        wire_pts, bulb_pts, bulb_cidx, bulb_ph = [], [], [], []
        for sw_i, (xa, xb) in enumerate(swags):
            xs_w = np.linspace(xa, xb, 38, dtype=np.float32)
            uw = (xs_w - xa) / (xb - xa)
            ys_w = 0.87 - 0.085 * np.sin(uw * math.pi)
            wire_pts.append(np.stack([xs_w, ys_w, np.full_like(xs_w, BACK_WALL_Z - 0.01)], axis=1))
            xs_b = np.linspace(xa + 0.04, xb - 0.04, 8, dtype=np.float32)
            ub = (xs_b - xa) / (xb - xa)
            ys_b = 0.86 - 0.085 * np.sin(ub * math.pi)
            bulb_pts.append(np.stack([xs_b, ys_b, np.full_like(xs_b, BACK_WALL_Z - 0.02)], axis=1))
            bulb_cidx.append(np.arange(len(xs_b), dtype=np.int32) + sw_i * 3)
            bulb_ph.append(rng.uniform(0.0, math.tau, len(xs_b)).astype(np.float32))
        self.wire_pos = np.concatenate(wire_pts, axis=0)
        self.wire_nrm = np.tile(np.array([0.0, -0.3, 0.9], dtype=np.float32), (len(self.wire_pos), 1))
        self.wall_bulb_pos = np.concatenate(bulb_pts, axis=0)
        self.wall_bulb_cidx = np.concatenate(bulb_cidx, axis=0)
        self.wall_bulb_phase = np.concatenate(bulb_ph, axis=0)

        # 3-Part Triptych Snowy Window (big center + small left & right)
        win_segs = []
        for x1, x2, y1, y2, has_vmul in WINDOW_PANELS:
            win_segs.extend([
                ((x1, y1, WIN_Z), (x2, y1, WIN_Z), (0, -0.2, -1)),
                ((x1, y2, WIN_Z), (x2, y2, WIN_Z), (0, 0.2, -1)),
                ((x1, y1, WIN_Z), (x1, y2, WIN_Z), (0.2, 0, -1)),
                ((x2, y1, WIN_Z), (x2, y2, WIN_Z), (-0.2, 0, -1)),
                ((x1, WIN_MY, WIN_Z), (x2, WIN_MY, WIN_Z), (0, 0, -1)),
                ((x1 - 0.02, y1 - 0.022, WIN_Z - 0.04), (x2 + 0.02, y1 - 0.022, WIN_Z - 0.04), (0, 0.7, -0.7)),
            ])
            if has_vmul:
                win_segs.append(((0.5 * (x1 + x2), y1, WIN_Z), (0.5 * (x1 + x2), y2, WIN_Z), (0, 0, -1)))
        wp, wn = zip(*[_line_pts(a, b, density=150.0, normal=n) for a, b, n in win_segs])
        self.win_frame_pos = np.concatenate(wp, axis=0)
        self.win_frame_nrm = np.concatenate(wn, axis=0)

        # Crescent moon in upper-right pane of center window
        mx, my = -0.27, 0.62
        cand_x = rng.uniform(mx - 0.07, mx + 0.07, 650).astype(np.float32)
        cand_y = rng.uniform(my - 0.07, my + 0.07, 650).astype(np.float32)
        in_main = ((cand_x - mx) ** 2 + (cand_y - my) ** 2) <= 0.058**2
        in_cut = ((cand_x - (mx + 0.025)) ** 2 + (cand_y - (my + 0.012)) ** 2) <= 0.050**2
        keep = in_main & (~in_cut)
        self.moon_pos = np.stack([cand_x[keep], cand_y[keep], np.full(keep.sum(), WIN_Z + 0.025, dtype=np.float32)], axis=1)
        self.moon_nrm = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (int(keep.sum()), 1))

        # Stars, settled snow & falling snow across all 3 windows
        star_pos_l, star_ph_l = [], []
        drift_pos_l = []
        sx_l, sy_l, bx1_l, bx2_l, by1_l, by2_l, has_v_l = [], [], [], [], [], [], []
        for x1, x2, y1, y2, has_v in WINDOW_PANELS:
            ns = 18 if has_v else 8
            star_pos_l.append(np.stack([
                rng.uniform(x1 + 0.03, x2 - 0.03, ns).astype(np.float32),
                rng.uniform(y1 + 0.04, y2 - 0.03, ns).astype(np.float32),
                np.full(ns, WIN_Z + 0.03, dtype=np.float32),
            ], axis=1))
            star_ph_l.append(rng.uniform(0.0, math.tau, ns).astype(np.float32))

            nd = 75 if has_v else 35
            drift_pos_l.append(np.stack([
                rng.uniform(x1 + 0.02, x2 - 0.02, nd).astype(np.float32),
                rng.uniform(y1 + 0.01, y1 + 0.026, nd).astype(np.float32),
                np.full(nd, WIN_Z + 0.018, dtype=np.float32),
            ], axis=1))

            nf = 44 if has_v else 20
            sx_l.append(rng.uniform(x1 + 0.02, x2 - 0.02, nf).astype(np.float32))
            sy_l.append(rng.uniform(y1 + 0.02, y2 - 0.02, nf).astype(np.float32))
            bx1_l.append(np.full(nf, x1 + 0.02, dtype=np.float32))
            bx2_l.append(np.full(nf, x2 - 0.02, dtype=np.float32))
            by1_l.append(np.full(nf, y1 + 0.02, dtype=np.float32))
            by2_l.append(np.full(nf, y2 - 0.02, dtype=np.float32))
            has_v_l.append(np.full(nf, has_v, dtype=bool))

        self.star_pos = np.concatenate(star_pos_l, axis=0)
        self.star_phase = np.concatenate(star_ph_l, axis=0)
        self.win_snow_drift = np.concatenate(drift_pos_l, axis=0)

        self.snow_x = np.concatenate(sx_l)
        self.snow_y = np.concatenate(sy_l)
        self.snow_x1 = np.concatenate(bx1_l)
        self.snow_x2 = np.concatenate(bx2_l)
        self.snow_y1 = np.concatenate(by1_l)
        self.snow_y2 = np.concatenate(by2_l)
        self.snow_has_vmul = np.concatenate(has_v_l)
        n_snow = len(self.snow_x)
        self.snow_speed = rng.uniform(0.13, 0.27, n_snow).astype(np.float32)
        self.snow_drift = rng.uniform(-0.03, 0.04, n_snow).astype(np.float32)
        self.snow_phase = rng.uniform(0.0, math.tau, n_snow).astype(np.float32)
        self.snow_emit = rng.uniform(0.58, 0.92, n_snow).astype(np.float32)

    def _apply_room_glow(
        self,
        pos: np.ndarray,
        base_col: np.ndarray,
        palette: ThemePalette,
        tree_glow: float,
        fire_flicker: float,
    ) -> np.ndarray:
        if len(pos) == 0:
            return base_col
        out = base_col
        if tree_glow > 0.0:
            dx = (pos[:, 0] - TREE_X) * 0.85
            dy = (pos[:, 1] - (FLOOR_Y + 0.62)) * 0.95
            dz = (pos[:, 2] - TREE_Z) * 0.85
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            w = np.clip(1.0 - dist / 1.70, 0.0, 1.0) ** 2 * (0.40 * tree_glow)
            warm_t = np.array(palette.garland, dtype=np.float32)[None, :]
            out = out * (1.0 - w[:, None]) + warm_t * w[:, None]
        if fire_flicker > 0.0:
            dx = (pos[:, 0] - FIRE_X) * 0.95
            dy = (pos[:, 1] - (FLOOR_Y + 0.25)) * 1.1
            dz = (pos[:, 2] - (BACK_WALL_Z - 0.15)) * 0.95
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            wf = np.clip(1.0 - dist / 1.15, 0.0, 1.0) ** 2 * (0.50 * fire_flicker)
            warm_f = np.array(palette.fire_glow, dtype=np.float32)[None, :]
            out = out * (1.0 - wf[:, None]) + warm_f * wf[:, None]
        return out

    def build_batch(
        self,
        palette: ThemePalette,
        t: float,
        room_reveal: float,
        snow_on: bool,
        tree_glow: float,
    ) -> PointBatch:
        if room_reveal <= 0.0:
            return empty_batch()

        max_y = FLOOR_Y + (0.95 - FLOOR_Y) * room_reveal + 0.05
        fire_flicker = (0.78 + 0.22 * math.sin(t * 7.3) * math.cos(t * 3.1)) * clamp01(room_reveal * 2.5)
        batches: list[PointBatch] = []

        def add_masked(
            pos: np.ndarray,
            nrm: np.ndarray,
            rgb: tuple[float, float, float],
            emit: float = 0.0,
            tint_glow: bool = True,
        ) -> None:
            mask = pos[:, 1] <= max_y
            if not np.any(mask):
                return
            p = pos[mask]
            n = nrm[mask]
            c = np.tile(np.array(rgb, dtype=np.float32), (len(p), 1))
            if tint_glow:
                c = self._apply_room_glow(p, c, palette, tree_glow, fire_flicker)
            e = np.full(len(p), emit, dtype=np.float32)
            batches.append(PointBatch(p, n, c, e))

        add_masked(self.floor_fill_pos, self.floor_fill_nrm, palette.floor_fill)
        add_masked(self.floor_line_pos, self.floor_line_nrm, palette.floor_line)

        rug_mask = self.rug_pos[:, 1] <= max_y
        if np.any(rug_mask):
            rp = self.rug_pos[rug_mask]
            rn = self.rug_nrm[rug_mask]
            rb = self.rug_is_border[rug_mask]
            rc = np.where(
                rb[:, None],
                np.array(palette.rug_outer, dtype=np.float32),
                np.array(palette.rug_inner, dtype=np.float32),
            )
            rc = self._apply_room_glow(rp, rc, palette, tree_glow, fire_flicker)
            batches.append(PointBatch(rp, rn, rc, np.zeros(len(rp), dtype=np.float32)))

        add_masked(self.wall_fill_pos, self.wall_fill_nrm, palette.wall_fill)
        add_masked(self.wall_line_pos, self.wall_line_nrm, palette.wall_line)

        # Fireplace bricks, mantel, wreath, stockings, logs & flickering 3D fire
        add_masked(self.fp_brick_fill_pos, self.fp_brick_fill_nrm, tuple(c * 0.58 for c in palette.hearth_brick))
        add_masked(self.fp_brick_pos, self.fp_brick_nrm, palette.hearth_brick)
        add_masked(self.fp_mantel_pos, self.fp_mantel_nrm, palette.hearth_mantel)
        add_masked(self.fp_log_pos, self.fp_log_nrm, (0.28, 0.14, 0.08), tint_glow=False)

        wm = self.wreath_pos[:, 1] <= max_y
        if np.any(wm):
            wp_m = self.wreath_pos[wm]
            wn_m = self.wreath_nrm[wm]
            wb_m = self.wreath_is_bow[wm]
            wc = np.where(
                wb_m[:, None],
                np.array((0.92, 0.16, 0.20), dtype=np.float32),
                np.array(palette.pine_bright, dtype=np.float32) * 0.78,
            )
            batches.append(PointBatch(wp_m, wn_m, wc, np.where(wb_m, 0.25, 0.08).astype(np.float32)))

        sm = self.stocking_pos[:, 1] <= max_y
        if np.any(sm):
            batches.append(
                PointBatch(
                    self.stocking_pos[sm],
                    self.stocking_nrm[sm],
                    self.stocking_col[sm],
                    np.full(int(sm.sum()), 0.16, dtype=np.float32),
                )
            )

        # Animated dancing flames inside firebox (natural tapered flame silhouette)
        if max_y > FLOOR_Y + 0.06:
            tongue = 0.72 + 0.28 * np.sin(self.fire_x0 * 34.0 + t * 5.2 + self.fire_phase)
            u_eff = self.fire_u * tongue
            fy = FLOOR_Y + 0.065 + u_eff * 0.30
            taper = np.clip(1.0 - (self.fire_u ** 0.78), 0.12, 1.0)
            sway = 0.034 * np.sin(t * self.fire_freq + self.fire_phase + self.fire_u * 4.5) * self.fire_u
            fx = FIRE_X + self.fire_x0 * 1.15 * taper + sway
            fz = np.full_like(fy, BACK_WALL_Z - 0.06)
            fm = fy <= max_y
            if np.any(fm):
                u_f = self.fire_u[fm]
                fp_pos = np.stack([fx[fm], fy[fm], fz[fm]], axis=1)
                fp_nrm = np.tile(np.array([0.0, 0.3, -0.95], dtype=np.float32), (len(fp_pos), 1))
                # Deep orange-red core -> warm amber -> bright golden flame tips
                col_bot = np.array((1.00, 0.30, 0.06), dtype=np.float32)[None, :]
                col_top = np.array((1.00, 0.90, 0.28), dtype=np.float32)[None, :]
                fc = col_bot * (1.0 - u_f[:, None]) + col_top * u_f[:, None]
                fe = (1.15 - 0.35 * u_f).astype(np.float32)
                batches.append(PointBatch(fp_pos, fp_nrm, fc, fe))

            # Subtle rising embers
            su = (self.fp_spark_t0 + t * self.fp_spark_spd) % 1.0
            sy_s = FLOOR_Y + 0.14 + su * 0.28
            sx_s = FIRE_X + self.fp_spark_x0 + 0.025 * np.sin(t * 4.5 + su * 6.0)
            sem = sy_s <= max_y
            if np.any(sem):
                sp_pos = np.stack([sx_s[sem], sy_s[sem], np.full(int(sem.sum()), BACK_WALL_Z - 0.07, dtype=np.float32)], axis=1)
                sp_nrm = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(sp_pos), 1))
                sp_col = np.tile(np.array((1.00, 0.85, 0.35), dtype=np.float32), (len(sp_pos), 1))
                batches.append(PointBatch(sp_pos, sp_nrm, sp_col, (1.1 * (1.0 - su[sem])).astype(np.float32)))

        # Upper wall fairy string lights
        add_masked(self.wire_pos, self.wire_nrm, palette.wall_line, tint_glow=False)
        bm = self.wall_bulb_pos[:, 1] <= max_y
        if np.any(bm):
            bp = self.wall_bulb_pos[bm]
            bn = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(bp), 1))
            pal_l = np.array(palette.lights, dtype=np.float32)
            tw_b = 0.45 + 0.65 * (0.5 + 0.5 * np.sin(t * 3.2 + self.wall_bulb_phase[bm]))
            bc = pal_l[self.wall_bulb_cidx[bm] % len(palette.lights)] * tw_b[:, None]
            batches.append(PointBatch(bp, bn, bc.astype(np.float32), (0.6 + 0.6 * tw_b).astype(np.float32)))

        # Small "DEC 25" wall calendar frame
        add_masked(self.cal_frame_pos, self.cal_frame_nrm, palette.window_frame)

        # 3-Part Snowy Window
        add_masked(self.win_frame_pos, self.win_frame_nrm, palette.window_frame)
        add_masked(self.moon_pos, self.moon_nrm, palette.moon, emit=0.90, tint_glow=False)
        add_masked(
            self.win_snow_drift,
            np.tile([0.0, 0.0, -1.0], (len(self.win_snow_drift), 1)),
            (0.76, 0.85, 0.98),
            emit=0.22,
            tint_glow=False,
        )

        star_mask = self.star_pos[:, 1] <= max_y
        if np.any(star_mask):
            sp = self.star_pos[star_mask]
            tw = 0.45 + 0.55 * np.sin(t * 2.2 + self.star_phase[star_mask])
            sc = np.array([0.82, 0.90, 1.00], dtype=np.float32)[None, :] * tw[:, None]
            sn = np.tile(np.array([0.0, 0.2, 0.8], dtype=np.float32), (len(sp), 1))
            batches.append(PointBatch(sp, sn, sc, tw.astype(np.float32) * 0.55))

        if snow_on:
            sy = self.snow_y1 + np.mod(self.snow_y - self.snow_y1 - t * self.snow_speed, (self.snow_y2 - self.snow_y1))
            sx = self.snow_x1 + np.mod(
                self.snow_x - self.snow_x1 + t * self.snow_drift + 0.020 * np.sin(t * 1.4 + self.snow_phase),
                (self.snow_x2 - self.snow_x1),
            )
            clear_vmul = (~self.snow_has_vmul) | (np.abs(sx - WIN_MX) > 0.020)
            vis = (sy <= max_y) & clear_vmul & (np.abs(sy - WIN_MY) > 0.020)
            if np.any(vis):
                snp = np.stack([sx[vis], sy[vis], np.full(int(vis.sum()), WIN_Z + 0.015, dtype=np.float32)], axis=1)
                snn = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(snp), 1))
                snc = np.tile(np.array([0.92, 0.96, 1.00], dtype=np.float32), (len(snp), 1))
                batches.append(PointBatch(snp, snn, snc, self.snow_emit[vis]))

        return concat_batches(batches)


class TreeModel:
    """High-density volumetric 3D spinning Christmas tree (~50,000 points)."""

    TRUNK_TOP = 0.31

    @staticmethod
    def envelope(y: float) -> float:
        """Tiered conifer envelope in local y in [0.27, 1.46]."""
        u = clamp01((y - 0.27) / (1.46 - 0.27))
        base = 0.63 * (1.0 - u) ** 0.84 + 0.025
        tier_phase = (u * 4.0) % 1.0
        tier_flare = 0.76 + 0.28 * (1.0 - tier_phase)
        if u > 0.90:
            tier_flare = 1.0 - (u - 0.90) * 3.5
        return max(0.02, base * tier_flare)

    def __init__(self, seed: int = 2026) -> None:
        rng = np.random.default_rng(seed)

        # 1. Trunk & red tree stand
        stand_pos, stand_nrm = _tube(rng, (0.0, 0.0, 0.0), (0.0, 0.075, 0.0), 0.145, 1100)
        trunk_pos, trunk_nrm = _tube(rng, (0.0, 0.06, 0.0), (0.0, self.TRUNK_TOP + 0.06, 0.0), 0.074, 2100)
        self.trunk_pos = np.concatenate([stand_pos, trunk_pos], axis=0)
        self.trunk_nrm = np.concatenate([stand_nrm, trunk_nrm], axis=0)
        self.trunk_is_stand = np.concatenate(
            [np.ones(len(stand_pos), dtype=bool), np.zeros(len(trunk_pos), dtype=bool)]
        )

        # 2. Dense inner pine core (~16,000 points)
        n_inner = 16000
        iy = rng.uniform(0.27, 1.42, n_inner).astype(np.float32)
        ith = rng.uniform(0.0, math.tau, n_inner).astype(np.float32)
        ir = np.array([self.envelope(float(yv)) for yv in iy], dtype=np.float32) * (
            0.35 + 0.52 * np.sqrt(rng.random(n_inner, dtype=np.float32))
        )
        self.inner_pos = np.stack([ir * np.cos(ith), iy, ir * np.sin(ith)], axis=1)
        self.inner_nrm = np.stack([np.cos(ith), np.full(n_inner, 0.25, dtype=np.float32), np.sin(ith)], axis=1)
        self.inner_nrm /= np.linalg.norm(self.inner_nrm, axis=1, keepdims=True) + 1e-8

        # 3. Dense drooping branch whorls & needles (~31,000 points)
        needle_pos_list: list[np.ndarray] = []
        needle_nrm_list: list[np.ndarray] = []
        needle_shade_list: list[np.ndarray] = []
        tips_list: list[np.ndarray] = []

        for row_idx, y0 in enumerate(np.linspace(0.27, 1.42, 34)):
            u = (y0 - 0.27) / (1.42 - 0.27)
            env = self.envelope(float(y0))
            n_branches = int(10 + 14 * (1.0 - u))
            for b in range(n_branches):
                a = math.tau * (b + 0.45 * rng.random()) / n_branches + row_idx * 0.77
                b_len = env * rng.uniform(0.87, 1.05)
                n_pts = int(42 + 68 * b_len)
                s = rng.random(n_pts, dtype=np.float32) ** 0.72
                d = np.array([math.cos(a), 0.0, math.sin(a)], dtype=np.float32)
                side = np.array([-math.sin(a), 0.0, math.cos(a)], dtype=np.float32)
                spread = (0.018 + 0.052 * s) * rng.normal(size=n_pts).astype(np.float32)
                p = np.outer(s * b_len, d) + np.outer(spread, side)
                p[:, 1] = float(y0) - 0.08 * (s * b_len) ** 1.7 + 0.017 * rng.normal(size=n_pts).astype(np.float32)
                nrm = np.outer(np.full(n_pts, 0.85, dtype=np.float32), d) + np.outer(
                    rng.normal(scale=0.35, size=n_pts).astype(np.float32), side
                )
                nrm[:, 1] += 0.35
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
                shade = np.clip(0.15 + 0.84 * s + 0.22 * (1.0 - u) * 0.2, 0.0, 1.0).astype(np.float32)
                needle_pos_list.append(p.astype(np.float32))
                needle_nrm_list.append(nrm.astype(np.float32))
                needle_shade_list.append(shade)
                tips_list.append(p[int(np.argmax(s))])

        self.needle_pos = np.concatenate(needle_pos_list, axis=0)
        self.needle_nrm = np.concatenate(needle_nrm_list, axis=0)
        self.needle_shade = np.concatenate(needle_shade_list, axis=0)
        tips = np.array(tips_list, dtype=np.float32)
        tips = tips[(tips[:, 1] > 0.31) & (tips[:, 1] < 1.38)]

        # 4. Spiraling gold garland with swags (~4,600 points)
        n_gar = 4600
        v = np.linspace(0.0, 1.0, n_gar, dtype=np.float32)
        th = math.tau * 4.5 * v
        sag = 0.05 * np.sin(math.pi * ((th / math.tau * 4.0) % 1.0))
        gy = 0.32 + (1.37 - 0.32) * v - sag
        gr = np.array([self.envelope(float(yv)) for yv in gy], dtype=np.float32) * 1.01
        gx = gr * np.cos(th) + 0.008 * rng.normal(size=n_gar).astype(np.float32)
        gz = gr * np.sin(th) + 0.008 * rng.normal(size=n_gar).astype(np.float32)
        self.garland_pos = np.stack([gx, gy, gz], axis=1).astype(np.float32)
        self.garland_nrm = np.stack([np.cos(th), np.full(n_gar, 0.2, dtype=np.float32), np.sin(th)], axis=1).astype(np.float32)
        self.garland_nrm /= np.linalg.norm(self.garland_nrm, axis=1, keepdims=True) + 1e-8
        self.garland_v = v

        # 5. Fuller 3D spherical baubles (~56 spheres x 54 points)
        pick_b = rng.choice(len(tips), min(56, len(tips)), replace=False)
        sph = _fib_sphere(54)
        b_pos, b_nrm, b_cidx, b_y = [], [], [], []
        for idx, ti in enumerate(pick_b):
            center = tips[ti]
            pts = center[None, :] + 0.038 * sph
            b_pos.append(pts.astype(np.float32))
            b_nrm.append(sph.astype(np.float32))
            b_cidx.append(np.full(len(sph), idx, dtype=np.int32))
            b_y.append(np.full(len(sph), center[1], dtype=np.float32))
        self.bauble_pos = np.concatenate(b_pos, axis=0)
        self.bauble_nrm = np.concatenate(b_nrm, axis=0)
        self.bauble_cidx = np.concatenate(b_cidx, axis=0)
        self.bauble_y = np.concatenate(b_y, axis=0)

        # 6. Twinkling fairy lights (~190 lights)
        avail = np.setdiff1d(np.arange(len(tips)), pick_b)
        pick_l = rng.choice(avail, min(190, len(avail)), replace=False)
        self.light_pos = tips[pick_l].astype(np.float32)
        self.light_nrm = np.tile(np.array([0.0, 0.3, -0.95], dtype=np.float32), (len(self.light_pos), 1))
        self.light_phase = rng.uniform(0.0, math.tau, len(self.light_pos)).astype(np.float32)
        self.light_cidx = rng.integers(0, 100, len(self.light_pos), dtype=np.int32)
        self.light_angle = np.arctan2(self.light_pos[:, 2], self.light_pos[:, 0]).astype(np.float32)

        # 7. Crisp 5-pointed 3D star at crown (y = 1.54)
        n_star = 1800
        sx = rng.uniform(-0.18, 0.18, n_star).astype(np.float32)
        sy = rng.uniform(-0.18, 0.18, n_star).astype(np.float32)
        ang_s = np.arctan2(sx, sy)
        f_star = np.abs(((ang_s / math.tau * 5.0) % 1.0) - 0.5) * 2.0
        max_r = 0.044 + 0.128 * (f_star ** 1.4)
        rad_s = np.hypot(sx, sy)
        keep_s = rad_s <= max_r
        sx, sy, rad_s = sx[keep_s], sy[keep_s], rad_s[keep_s]
        sz = (0.034 * (1.0 - np.clip(rad_s / 0.17, 0.0, 1.0)) * rng.choice([-1.0, 1.0], size=len(sx))).astype(np.float32)
        self.star_pos = np.stack([sx, 1.54 + sy, sz], axis=1).astype(np.float32)
        self.star_nrm = np.stack([sx * 2.0, sy * 2.0, np.where(sz >= 0, 1.0, -1.0)], axis=1).astype(np.float32)
        self.star_nrm /= np.linalg.norm(self.star_nrm, axis=1, keepdims=True) + 1e-8

    def build_batch(
        self,
        palette: ThemePalette,
        t: float,
        trunk_grow: float,
        tree_grow: float,
        garland_grow: float,
        ornament_grow: float,
        lights_power: float,
        star_grow: float,
        spin_angle: float,
        light_mode: str,
    ) -> PointBatch:
        if trunk_grow <= 0.0:
            return empty_batch()

        R = rot_y(spin_angle)
        center = np.array([TREE_X, FLOOR_Y, TREE_Z], dtype=np.float32)
        batches: list[PointBatch] = []

        def to_world(local_p: np.ndarray, local_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (local_p @ R.T) + center[None, :], (local_n @ R.T)

        trunk_cut = (self.TRUNK_TOP + 0.06) * trunk_grow
        tm = self.trunk_pos[:, 1] <= trunk_cut
        if np.any(tm):
            tp, tn = to_world(self.trunk_pos[tm], self.trunk_nrm[tm])
            is_s = self.trunk_is_stand[tm]
            tc = np.where(
                is_s[:, None],
                np.array(palette.rug_outer, dtype=np.float32),
                np.array(palette.trunk, dtype=np.float32),
            )
            te = np.where(is_s, 0.15, 0.12).astype(np.float32)
            batches.append(PointBatch(tp, tn, tc, te))

        if tree_grow <= 0.0:
            return concat_batches(batches)

        tree_cut = 0.27 + (1.46 - 0.27) * tree_grow

        im = self.inner_pos[:, 1] <= tree_cut
        if np.any(im):
            ip, inn = to_world(self.inner_pos[im], self.inner_nrm[im])
            ic = np.tile(np.array(palette.pine_dark, dtype=np.float32) * 0.72, (len(ip), 1))
            batches.append(PointBatch(ip, inn, ic, np.zeros(len(ip), dtype=np.float32)))

        nm = self.needle_pos[:, 1] <= tree_cut
        if np.any(nm):
            np_pos, np_nrm = to_world(self.needle_pos[nm], self.needle_nrm[nm])
            s = self.needle_shade[nm][:, None]
            c_dark = np.array(palette.pine_dark, dtype=np.float32)[None, :]
            c_bright = np.array(palette.pine_bright, dtype=np.float32)[None, :]
            nc = c_dark * (1.0 - s) + c_bright * s
            batches.append(PointBatch(np_pos, np_nrm, nc, np.zeros(len(np_pos), dtype=np.float32)))

        if garland_grow > 0.0:
            gm = self.garland_v <= garland_grow
            if np.any(gm):
                gp, gn = to_world(self.garland_pos[gm], self.garland_nrm[gm])
                shimmer = 0.85 + 0.25 * np.sin(t * 2.4 + self.garland_v[gm] * 18.0)
                gc = np.array(palette.garland, dtype=np.float32)[None, :] * shimmer[:, None]
                batches.append(PointBatch(gp, gn, gc, np.full(len(gp), 0.26, dtype=np.float32)))

        if ornament_grow > 0.0:
            orn_cut = 0.29 + (1.40 - 0.29) * ornament_grow
            bm = self.bauble_y <= orn_cut
            if np.any(bm):
                bp, bn = to_world(self.bauble_pos[bm], self.bauble_nrm[bm])
                pal_arr = np.array(palette.baubles, dtype=np.float32)
                bc = pal_arr[self.bauble_cidx[bm] % len(palette.baubles)]
                batches.append(PointBatch(bp, bn, bc, np.full(len(bp), 0.28, dtype=np.float32)))

        if lights_power > 0.0:
            l_cut = 0.29 + (1.42 - 0.29) * lights_power
            lm = self.light_pos[:, 1] <= l_cut
            if np.any(lm):
                lp, ln = to_world(self.light_pos[lm], self.light_nrm[lm])
                l_cidx = self.light_cidx[lm]
                l_ph = self.light_phase[lm]
                l_y = self.light_pos[lm, 1]
                l_ang = self.light_angle[lm]
                pal_l = np.array(palette.lights, dtype=np.float32)

                eff_mode = light_mode
                if eff_mode == "auto":
                    cycle = ("twinkle", "chase", "sparkle", "bands")
                    eff_mode = cycle[int(t // 8.0) % len(cycle)]

                if eff_mode == "twinkle":
                    gain = 0.35 + 0.95 * (0.5 + 0.5 * np.sin(t * 4.5 + l_ph))
                    lc = pal_l[l_cidx % len(palette.lights)] * gain[:, None]
                    emit = 0.45 + 0.85 * gain
                elif eff_mode == "chase":
                    wave = np.sin(l_y * 14.0 + l_ang * 2.0 - t * 6.0)
                    gain = np.where(wave > 0.45, 1.35, np.where(wave > -0.2, 0.72, 0.25))
                    lc = pal_l[(l_cidx + int(t * 0.4)) % len(palette.lights)] * gain[:, None]
                    emit = np.where(wave > 0.45, 1.30, 0.35)
                elif eff_mode == "sparkle":
                    flash = np.sin(t * 9.5 + l_ph * 4.3) > 0.84
                    base_g = 0.45 + 0.30 * np.sin(t * 2.2 + l_ph)
                    lc = np.where(
                        flash[:, None],
                        np.array([1.0, 0.98, 0.90], dtype=np.float32)[None, :],
                        pal_l[l_cidx % len(palette.lights)] * base_g[:, None],
                    )
                    emit = np.where(flash, 1.45, 0.40)
                else:  # bands
                    band = np.floor((l_y - 0.29) * 6.0 - t * 1.4).astype(np.int32) % len(palette.lights)
                    pulse = 0.78 + 0.35 * np.sin(t * 3.2 + l_y * 6.0)
                    lc = pal_l[band] * pulse[:, None]
                    emit = 0.70 + 0.50 * pulse

                batches.append(PointBatch(lp, ln, lc.astype(np.float32), emit.astype(np.float32)))

        if star_grow > 0.0:
            star_scale = 0.45 + 0.55 * star_grow
            sp_local = self.star_pos.copy()
            sp_local[:, 1] = 1.54 + (sp_local[:, 1] - 1.54) * star_scale - (1.0 - star_grow) * 0.12
            sp_local[:, 0] *= star_scale
            sp_local[:, 2] *= star_scale
            star_R = rot_y(spin_angle * 0.65)
            sp = (sp_local @ star_R.T) + center[None, :]
            sn = self.star_nrm @ star_R.T
            pulse = (0.85 + 0.25 * math.sin(t * 3.4)) * star_grow
            sc = np.tile(np.array(palette.star, dtype=np.float32) * pulse, (len(sp), 1))
            batches.append(PointBatch(sp, sn, sc, np.full(len(sp), 1.20 * star_grow, dtype=np.float32)))

        return concat_batches(batches)


@dataclass
class GiftBox:
    pos: np.ndarray
    nrm: np.ndarray
    is_ribbon: np.ndarray
    wrap_idx: int
    world_xy: tuple[float, float]
    yaw: float
    appear_delay: float


class GiftsModel:
    """Five high-density 3D gift boxes with crossed ribbons & bows around the tree."""

    def __init__(self, seed: int = 99) -> None:
        rng = np.random.default_rng(seed)
        self.boxes: list[GiftBox] = []
        for wx, wz, w, h, d, yaw, widx, delay in GIFT_SPECS:
            n_box = 2400
            ax = rng.integers(0, 3, n_box)
            sgn = rng.choice([-1.0, 1.0], n_box).astype(np.float32)
            u = rng.uniform(-1.0, 1.0, (n_box, 2)).astype(np.float32)
            p = np.zeros((n_box, 3), dtype=np.float32)
            nrm = np.zeros((n_box, 3), dtype=np.float32)
            half = np.array([w * 0.5, h * 0.5, d * 0.5], dtype=np.float32)
            for k in range(n_box):
                a = ax[k]
                b, c = (a + 1) % 3, (a + 2) % 3
                p[k, a] = sgn[k] * half[a]
                p[k, b] = u[k, 0] * half[b]
                p[k, c] = u[k, 1] * half[c]
                nrm[k, a] = sgn[k]
            p[:, 1] += half[1]

            rib = (np.abs(p[:, 0]) < w * 0.14) | (np.abs(p[:, 2]) < d * 0.14)
            b1, n1 = _blob(rng, (-0.032, h + 0.024, 0.0), (0.040, 0.022, 0.020), 180, 0.6)
            b2, n2 = _blob(rng, (0.032, h + 0.024, 0.0), (0.040, 0.022, 0.020), 180, 0.6)
            all_p = np.concatenate([p, b1, b2], axis=0)
            all_n = np.concatenate([nrm, n1, n2], axis=0)
            all_rib = np.concatenate([rib, np.ones(len(b1) + len(b2), dtype=bool)])
            self.boxes.append(GiftBox(all_p, all_n, all_rib, widx, (wx, wz), yaw, delay))

    def build_batch(
        self,
        palette: ThemePalette,
        gift_progress: float,
        ease_fn,
    ) -> PointBatch:
        if gift_progress <= 0.0:
            return empty_batch()

        batches: list[PointBatch] = []
        for box in self.boxes:
            local_u = clamp01((gift_progress - box.appear_delay) / 0.30)
            if local_u <= 0.0:
                continue
            sc = ease_fn(local_u)
            R = rot_y(box.yaw)
            p_scaled = box.pos * sc
            world_p = (p_scaled @ R.T) + np.array([box.world_xy[0], FLOOR_Y, box.world_xy[1]], dtype=np.float32)[None, :]
            world_n = box.nrm @ R.T
            wrap_c = np.array(palette.gift_wraps[box.wrap_idx % len(palette.gift_wraps)], dtype=np.float32)
            rib_c = np.array(palette.gift_ribbons[box.wrap_idx % len(palette.gift_ribbons)], dtype=np.float32)
            col = np.where(box.is_ribbon[:, None], rib_c[None, :], wrap_c[None, :])
            emit = np.where(box.is_ribbon, 0.50, 0.20).astype(np.float32)
            batches.append(PointBatch(world_p, world_n, col, emit))
        return concat_batches(batches)


class SantaRig:
    """Dense precomputed articulated 3D Santa (~11,500 points) with expressive face details."""

    HIP_Y = 0.28

    def __init__(self, seed: int = 321) -> None:
        rng = np.random.default_rng(seed)

        RED = (0.88, 0.14, 0.16)
        SKIN = (0.96, 0.78, 0.64)
        CHEEK = (0.98, 0.52, 0.48)
        EYE = (0.06, 0.06, 0.09)
        WHITE = (0.96, 0.97, 0.99)
        BELT = (0.14, 0.12, 0.14)
        BUCKLE = (1.00, 0.84, 0.26)
        BOOT = (0.16, 0.14, 0.16)
        SACK = (0.58, 0.39, 0.22)

        def make_batch(items: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]]) -> PointBatch:
            return concat_batches(
                [
                    PointBatch(
                        p.astype(np.float32),
                        n.astype(np.float32),
                        np.tile(np.array(col, dtype=np.float32), (len(p), 1)),
                        np.full(len(p), emit, dtype=np.float32),
                    )
                    for p, n, col, emit in items
                ]
            )

        def build_leg(side_z: float) -> PointBatch:
            items = []
            p, n = _tube(rng, (0.0, 0.0, side_z), (0.0, -0.14, side_z), 0.052, 420)
            items.append((p, n, RED, 0.08))
            p, n = _tube(rng, (0.0, -0.14, side_z), (0.0, -0.25, side_z), 0.046, 380)
            items.append((p, n, BOOT, 0.05))
            p, n = _blob(rng, (0.036, -0.258, side_z), (0.068, 0.030, 0.045), 280)
            items.append((p, n, BOOT, 0.05))
            return make_batch(items)

        def build_arm(side_z: float) -> PointBatch:
            items = []
            p, n = _tube(rng, (0.0, 0.0, side_z), (0.0, -0.22, side_z * 1.12), 0.044, 580)
            items.append((p, n, RED, 0.08))
            p, n = _blob(rng, (0.0, -0.21, side_z * 1.12), (0.048, 0.025, 0.048), 200)
            items.append((p, n, WHITE, 0.16))
            p, n = _blob(rng, (0.0, -0.25, side_z * 1.12), (0.038, 0.038, 0.038), 220)
            items.append((p, n, WHITE, 0.18))
            return make_batch(items)

        self.leg_l = build_leg(0.072)
        self.leg_r = build_leg(-0.072)
        self.arm_l = build_arm(0.165)
        self.arm_r = build_arm(-0.165)

        up_items = []
        p, n = _blob(rng, (0.015, 0.14, 0.0), (0.16, 0.17, 0.17), 3000, 0.50)
        up_items.append((p, n, RED, 0.08))
        p, n = _blob(rng, (0.01, 0.015, 0.0), (0.176, 0.032, 0.186), 780, 0.65)
        up_items.append((p, n, WHITE, 0.16))
        p, n = _tube(rng, (0.15, 0.02, 0.0), (0.13, 0.27, 0.0), 0.026, 360)
        up_items.append((p, n, WHITE, 0.16))
        p, n = _blob(rng, (0.015, 0.10, 0.0), (0.172, 0.026, 0.182), 680, 0.85)
        up_items.append((p, n, BELT, 0.02))
        p, n = _blob(rng, (0.176, 0.10, 0.0), (0.022, 0.032, 0.036), 240, 0.7)
        up_items.append((p, n, BUCKLE, 0.45))

        head_y = 0.36
        p, n = _blob(rng, (0.01, head_y, 0.0), (0.095, 0.098, 0.095), 1400, 0.55)
        up_items.append((p, n, SKIN, 0.14))
        # Rosy nose + dark expressive eyes
        p, n = _blob(rng, (0.106, head_y - 0.004, 0.0), (0.022, 0.020, 0.022), 140)
        up_items.append((p, n, CHEEK, 0.18))
        for ez in (-0.036, 0.036):
            p, n = _blob(rng, (0.098, head_y + 0.025, ez), (0.014, 0.014, 0.014), 70)
            up_items.append((p, n, EYE, 0.0))
        # Fluffy beard & mustache
        p, n = _blob(rng, (0.076, head_y - 0.088, 0.0), (0.082, 0.096, 0.095), 1250, 0.42)
        up_items.append((p, n, WHITE, 0.18))
        p, n = _blob(rng, (0.106, head_y - 0.032, 0.0), (0.036, 0.022, 0.056), 300, 0.5)
        up_items.append((p, n, WHITE, 0.18))
        # Hat brim, red cone, pom-pom
        p, n = _blob(rng, (0.005, head_y + 0.068, 0.0), (0.108, 0.026, 0.108), 540, 0.7)
        up_items.append((p, n, WHITE, 0.18))
        hat_tip = (-0.06, head_y + 0.22, -0.015)
        p, n = _tube(rng, (0.0, head_y + 0.09, 0.0), hat_tip, 0.058, 650)
        taper = np.clip(1.0 - (p[:, 1] - (head_y + 0.09)) / 0.16, 0.25, 1.0)
        p[:, 0] = hat_tip[0] * (1.0 - taper) + p[:, 0] * taper
        p[:, 2] = hat_tip[2] * (1.0 - taper) + p[:, 2] * taper
        up_items.append((p, n, RED, 0.08))
        p, n = _blob(rng, hat_tip, (0.034, 0.034, 0.034), 260, 0.6)
        up_items.append((p, n, WHITE, 0.24))
        self.upper = make_batch(up_items)

        p, n = _blob(rng, (0.0, 0.0, 0.0), (0.135, 0.175, 0.125), 1250, 0.55)
        self.sack = make_batch([(p, n, SACK, 0.05)])

    @staticmethod
    def _xform(batch: PointBatch, R: np.ndarray, offset: tuple[float, float, float]) -> PointBatch:
        off = np.array(offset, dtype=np.float32)[None, :]
        return PointBatch(
            pos=(batch.pos @ R.T) + off,
            nrm=(batch.nrm @ R.T),
            col=batch.col,
            emit=batch.emit,
        )

    def build_batch(
        self,
        t: float,
        x_pos: float,
        z_pos: float,
        yaw: float,
        walk_phase: float,
        stride_amp: float,
        bend: float,
        wave: float,
        laugh_bounce: float,
        sack_on_back: bool,
    ) -> tuple[PointBatch, np.ndarray]:
        hip_y = self.HIP_Y + laugh_bounce * 0.025
        swing = math.sin(walk_phase) * stride_amp

        leg_l = self._xform(self.leg_l, rot_z(-swing * 0.60), (0.0, hip_y, 0.0))
        leg_r = self._xform(self.leg_r, rot_z(swing * 0.60), (0.0, hip_y, 0.0))

        bend_R = rot_z(-bend * 0.42)
        upper = self._xform(self.upper, bend_R, (0.0, hip_y, 0.0))

        sh_r_local = np.array([0.015, 0.26, 0.0], dtype=np.float32) @ bend_R.T + np.array([0.0, hip_y, 0.0], dtype=np.float32)
        if wave > 0.02 or bend > 0.05:
            r_raise = wave * (1.85 + 0.40 * math.sin(t * 7.2)) + bend * 1.15
        else:
            r_raise = -swing * 0.65
        arm_r = self._xform(self.arm_r, rot_z(r_raise), tuple(sh_r_local))

        sh_l_local = np.array([0.01, 0.26, 0.0], dtype=np.float32) @ bend_R.T + np.array([0.0, hip_y, 0.0], dtype=np.float32)
        if sack_on_back:
            l_raise = -0.85 + swing * 0.15
        else:
            l_raise = wave * (1.15 + 0.30 * math.sin(t * 7.2 + 0.8))
        arm_l = self._xform(self.arm_l, rot_z(l_raise), tuple(sh_l_local))

        if sack_on_back:
            sack_off = np.array([-0.17, 0.22, 0.10], dtype=np.float32) @ bend_R.T + np.array([0.0, hip_y, 0.0], dtype=np.float32)
            sack_b = self._xform(self.sack, rot_z(-0.25), tuple(sack_off))
        else:
            sack_b = self._xform(self.sack, rot_z(0.15), (-0.23, 0.15, 0.12))

        local_all = concat_batches([leg_l, leg_r, upper, arm_r, arm_l, sack_b])

        R_world = rot_y(yaw)
        world_off = np.array([x_pos, FLOOR_Y, z_pos], dtype=np.float32)
        scale = 0.88
        wp = ((local_all.pos * scale) @ R_world.T) + world_off[None, :]
        wn = local_all.nrm @ R_world.T

        hand_local = np.array([0.0, -0.25, -0.18], dtype=np.float32) @ rot_z(r_raise).T + sh_r_local
        hand_world = ((hand_local * scale) @ R_world.T) + world_off
        return PointBatch(wp, wn, local_all.col, local_all.emit), hand_world


def build_conjure_sparkles(
    palette: ThemePalette,
    hand_world: np.ndarray,
    conjure_progress: float,
    t: float,
) -> PointBatch:
    """Golden magic sparkles arcing from Santa's hand toward the growing gifts."""
    if conjure_progress <= 0.0 or conjure_progress >= 1.0:
        return empty_batch()

    targets = np.array(
        [[wx, FLOOR_Y + h * 0.55, wz] for wx, wz, _, h, _, _, _, _ in GIFT_SPECS],
        dtype=np.float32,
    )
    rng = np.random.default_rng(909)
    n_per = 42
    pts, cols, emits = [], [], []
    for ti, tgt in enumerate(targets):
        u = (rng.random(n_per, dtype=np.float32) + t * 0.8 + ti * 0.17) % 1.0
        arc = np.sin(u * math.pi) * 0.28
        p = hand_world[None, :] * (1.0 - u[:, None]) + tgt[None, :] * u[:, None]
        p[:, 1] += arc
        p += rng.normal(scale=0.025, size=p.shape).astype(np.float32)
        c_gold = np.array(palette.star, dtype=np.float32)
        c_alt = np.array(palette.lights[ti % len(palette.lights)], dtype=np.float32)
        mix = rng.random(n_per, dtype=np.float32)[:, None]
        c = c_gold[None, :] * mix + c_alt[None, :] * (1.0 - mix)
        pts.append(p)
        cols.append(c)
        emits.append(np.full(n_per, 1.35 * math.sin(conjure_progress * math.pi), dtype=np.float32))

    all_p = np.concatenate(pts, axis=0)
    all_n = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(all_p), 1))
    return PointBatch(all_p, all_n, np.concatenate(cols, axis=0), np.concatenate(emits, axis=0))
