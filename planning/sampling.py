"""Sampling-based planners in continuous space (collision-checked against the grid): RRT and RRT*,
plus shortcut smoothing."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .grid import line_of_sight, path_length


@dataclass
class TreeResult:
    name: str
    path: list[tuple[float, float]] | None
    cost: float
    iterations: int
    nodes: int
    seconds: float
    edges: list[tuple[int, int]] = field(default_factory=list)
    points: np.ndarray | None = None
    first_solution_iter: int | None = None
    cost_history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None


def _free(grid, p) -> bool:
    y, x = int(round(p[0])), int(round(p[1]))
    H, W = grid.shape
    return 0 <= y < H and 0 <= x < W and not grid[y, x]


def _edge_free(grid, a, b) -> bool:
    return line_of_sight(grid, (a[0], a[1]), (b[0], b[1]))


def rrt(grid: np.ndarray, start, goal, star: bool = False, iters: int = 4000, step: float = 4.0,
        goal_bias: float = 0.05, goal_tol: float = 2.0, gamma: float = 30.0, seed: int = 0,
        stop_at_first: bool = False) -> TreeResult:
    """RRT (star=False) or RRT* (star=True: choose the best parent in a shrinking radius and rewire)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    H, W = grid.shape
    cap = iters + 2
    P = np.zeros((cap, 2))
    parent = np.full(cap, -1)
    cost = np.zeros(cap)
    P[0] = start
    n = 1
    goal = np.asarray(goal, float)
    best_goal, best_cost, first = -1, math.inf, None
    history = []
    for it in range(1, iters + 1):
        q = goal if rng.random() < goal_bias else np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
        d = np.hypot(*(P[:n] - q).T)
        i_near = int(np.argmin(d))
        dist = d[i_near]
        if dist < 1e-9:
            continue
        new = P[i_near] + (q - P[i_near]) * min(1.0, step / dist)
        if not _free(grid, new) or not _edge_free(grid, P[i_near], new):
            continue
        par, c_new = i_near, cost[i_near] + np.hypot(*(new - P[i_near]))
        if star:
            r = min(gamma * math.sqrt(math.log(n + 1) / (n + 1)), step * 3)
            nbr = np.nonzero(np.hypot(*(P[:n] - new).T) <= r)[0]
            for j in nbr:  # choose the cheapest collision-free parent
                cj = cost[j] + np.hypot(*(new - P[j]))
                if cj < c_new and _edge_free(grid, P[j], new):
                    par, c_new = j, cj
        P[n], parent[n], cost[n] = new, par, c_new
        k = n
        n += 1
        if star:
            for j in nbr:  # rewire neighbours through the new node if that is cheaper
                cj = c_new + np.hypot(*(P[j] - new))
                if cj + 1e-9 < cost[j] and _edge_free(grid, new, P[j]):
                    delta = cost[j] - cj
                    parent[j] = k
                    # propagate the improvement to descendants
                    stack = [j]
                    while stack:
                        u = stack.pop()
                        cost[u] -= delta
                        stack.extend(np.nonzero(parent[:n] == u)[0].tolist())
        if np.hypot(*(new - goal)) <= goal_tol and (best_goal < 0 or c_new < best_cost):
            best_goal, best_cost = k, c_new
            first = first or it
        if best_goal >= 0:
            best_cost = cost[best_goal]
            if it % 50 == 0:
                history.append((it, float(best_cost)))
            if stop_at_first and not star:
                break
    path = None
    if best_goal >= 0:
        path, u = [], best_goal
        while u != -1:
            path.append((float(P[u, 0]), float(P[u, 1])))
            u = parent[u]
        path.reverse()
        if np.hypot(*(np.array(path[-1]) - goal)) > 1e-9:
            path.append((float(goal[0]), float(goal[1])))
    edges = [(int(parent[i]), i) for i in range(1, n) if parent[i] >= 0]
    return TreeResult("RRT*" if star else "RRT", path, path_length(path) if path else math.inf, it, n,
                      time.perf_counter() - t0, edges, P[:n].copy(), first, history)


def shortcut(grid: np.ndarray, path, tries: int = 200, seed: int = 0):
    """Random shortcutting: repeatedly replace a sub-path with a straight segment when it is collision-free."""
    if not path or len(path) < 3:
        return path
    rng = np.random.default_rng(seed)
    pts = [tuple(p) for p in path]
    for _ in range(tries):
        if len(pts) < 3:
            break
        i, j = sorted(rng.choice(len(pts), 2, replace=False))
        if j - i < 2:
            continue
        if line_of_sight(grid, pts[i], pts[j]):
            pts = pts[:i + 1] + pts[j:]
    return pts
