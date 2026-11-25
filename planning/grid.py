"""Graph-search planners on an 8-connected grid: BFS, Dijkstra, A*, weighted A*, greedy best-first, Theta*.

All share one best-first search loop and differ only in priority and parent update, which is the
point: A* is Dijkstra plus a heuristic, greedy is A* without g, Theta* is A* with line-of-sight parents.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

SQRT2 = math.sqrt(2)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


@dataclass
class Result:
    name: str
    path: list[tuple[int, int]] | None
    cost: float
    expanded: int
    seconds: float
    order: list[tuple[int, int]] = field(default_factory=list)  # expansion order, for animations

    @property
    def found(self) -> bool:
        return self.path is not None


def octile(a, b) -> float:
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def euclid(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(path) -> float:
    return sum(euclid(a, b) for a, b in zip(path[:-1], path[1:])) if path else math.inf


def line_of_sight(grid: np.ndarray, a, b) -> bool:
    """Supercover-style check: sample the segment densely; any obstacle cell blocks it."""
    (y0, x0), (y1, x1) = a, b
    n = int(max(abs(y1 - y0), abs(x1 - x0)) * 2) + 1
    ys = np.rint(np.linspace(y0, y1, n + 1)).astype(int)
    xs = np.rint(np.linspace(x0, x1, n + 1)).astype(int)
    return not grid[ys, xs].any()


def _neighbors(grid, node):
    H, W = grid.shape
    y, x = node
    for dy, dx, c in MOVES:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx]:
            if dy and dx and (grid[y + dy, x] or grid[y, x + dx]):
                continue  # no cutting corners past an obstacle
            yield (ny, nx), c


def search(grid: np.ndarray, start, goal, name: str = "A*", w_h: float = 1.0, use_g: bool = True,
           any_angle: bool = False, unit_cost: bool = False, record: bool = False) -> Result:
    t0 = time.perf_counter()
    start, goal = tuple(start), tuple(goal)
    if grid[start] or grid[goal]:
        return Result(name, None, math.inf, 0, 0.0)
    g = {start: 0.0}
    parent = {start: start}
    h = (lambda n: 0.0) if w_h == 0 else (lambda n: w_h * (euclid(n, goal) if any_angle else octile(n, goal)))
    tie = 0
    openq = [(h(start), 0, start)]
    closed = set()
    order = []
    while openq:
        _, _, node = heapq.heappop(openq)
        if node in closed:
            continue
        closed.add(node)
        if record:
            order.append(node)
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            cost = path_length(path)  # always geometric length, so planners compare fairly
            return Result(name, path, cost, len(closed), time.perf_counter() - t0, order)
        for nb, c in _neighbors(grid, node):
            if nb in closed:
                continue
            c = 1.0 if unit_cost else c
            par, ng = node, g[node] + c
            if any_angle:  # Theta*: connect to the grandparent if it can see the neighbour
                pp = parent[node]
                if line_of_sight(grid, pp, nb):
                    alt = g[pp] + euclid(pp, nb)
                    if alt < ng:
                        par, ng = pp, alt
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = par
                tie += 1
                heapq.heappush(openq, ((ng if use_g else 0.0) + h(nb), tie, nb))
    return Result(name, None, math.inf, len(closed), time.perf_counter() - t0, order)


def bfs(grid, s, g, **kw):
    return search(grid, s, g, "BFS", w_h=0, unit_cost=True, **kw)


def dijkstra(grid, s, g, **kw):
    return search(grid, s, g, "Dijkstra", w_h=0, **kw)


def astar(grid, s, g, **kw):
    return search(grid, s, g, "A*", **kw)


def weighted_astar(grid, s, g, w=2.0, **kw):
    return search(grid, s, g, f"Weighted A* (w={w:g})", w_h=w, **kw)


def greedy(grid, s, g, **kw):
    return search(grid, s, g, "Greedy best-first", use_g=False, **kw)


def theta_star(grid, s, g, **kw):
    return search(grid, s, g, "Theta*", any_angle=True, **kw)
