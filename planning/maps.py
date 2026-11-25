"""Test maps: boolean occupancy grids (True = obstacle), indexed [row, col] = [y, x]."""
from __future__ import annotations

import numpy as np


def inflate(grid: np.ndarray, radius: int) -> np.ndarray:
    """Grow obstacles by `radius` cells (robot footprint), so the planner can treat the robot as a point."""
    if radius <= 0:
        return grid.copy()
    out = grid.copy()
    H, W = grid.shape
    ys, xs = np.nonzero(grid)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y2, x2 = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
                out[y2, x2] = True
    return out


def random_obstacles(size: int = 100, n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.zeros((size, size), bool)
    for _ in range(n):
        w, h = rng.integers(3, size // 6, 2)
        x, y = rng.integers(0, size - w), rng.integers(0, size - h)
        g[y:y + h, x:x + w] = True
    g[:6, :6] = g[-6:, -6:] = False  # keep start/goal corners clear
    return g


def maze(cells: int = 12, cell_px: int = 8, seed: int = 0) -> np.ndarray:
    """Perfect maze by randomized depth-first search (recursive backtracker)."""
    rng = np.random.default_rng(seed)
    size = cells * cell_px + 1
    g = np.ones((size, size), bool)
    seen = np.zeros((cells, cells), bool)
    stack = [(0, 0)]
    seen[0, 0] = True

    def carve(cy, cx):
        g[cy * cell_px + 1:(cy + 1) * cell_px, cx * cell_px + 1:(cx + 1) * cell_px] = False

    carve(0, 0)
    while stack:
        cy, cx = stack[-1]
        nbrs = [(cy + dy, cx + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cy + dy < cells and 0 <= cx + dx < cells and not seen[cy + dy, cx + dx]]
        if not nbrs:
            stack.pop()
            continue
        ny, nx = nbrs[rng.integers(len(nbrs))]
        seen[ny, nx] = True
        carve(ny, nx)
        # knock down the wall between the two cells
        y0, y1 = sorted((cy, ny))
        x0, x1 = sorted((cx, nx))
        g[y0 * cell_px + 1:(y1 + 1) * cell_px, x0 * cell_px + 1:(x1 + 1) * cell_px] = False
        stack.append((ny, nx))
    return g


def rooms(size: int = 100) -> np.ndarray:
    """Office floor: rooms joined by narrow doors (a classic hard case for sampling planners)."""
    g = np.zeros((size, size), bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    for x in (33, 66):
        g[:, x:x + 2] = True
    for y in (50,):
        g[y:y + 2, :] = True
    for (y0, y1, x) in [(20, 26, 33), (70, 76, 33), (12, 18, 66), (80, 86, 66)]:
        g[y0:y1, x:x + 2] = False
    for (x0, x1) in [(14, 20), (46, 52), (84, 90)]:
        g[50:52, x0:x1] = False
    return g
