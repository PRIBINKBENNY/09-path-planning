import math

import numpy as np
import pytest

from planning import (astar, bfs, dijkstra, greedy, inflate, line_of_sight, maze, octile, path_length,
                      random_obstacles, rooms, rrt, shortcut, theta_star, weighted_astar)

MAPS = {f"random{s}": random_obstacles(seed=s) for s in range(4)} | {"maze": maze(seed=1), "rooms": rooms()}


def endpoints(g):
    H, W = g.shape
    return (3, 3) if not g[3, 3] else (4, 4), (H - 4, W - 4) if not g[H - 4, W - 4] else (H - 5, W - 5)


def valid_grid_path(g, path):
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        assert max(abs(y1 - y0), abs(x1 - x0)) == 1          # 8-connected steps
        assert not g[y1, x1]
        if y1 != y0 and x1 != x0:
            assert not g[y1, x0] and not g[y0, x1]               # no corner cutting


@pytest.mark.parametrize("name", MAPS)
def test_astar_is_optimal_and_expands_less(name):
    g = MAPS[name]
    s, t = endpoints(g)
    d, a = dijkstra(g, s, t), astar(g, s, t)
    assert d.found and a.found
    assert a.cost == pytest.approx(d.cost)
    assert a.expanded <= d.expanded
    valid_grid_path(g, a.path)
    assert a.path[0] == s and a.path[-1] == t


@pytest.mark.parametrize("name", MAPS)
def test_suboptimal_planners_are_bounded(name):
    g = MAPS[name]
    s, t = endpoints(g)
    opt = dijkstra(g, s, t).cost
    w = weighted_astar(g, s, t, 2.0)
    assert opt <= w.cost <= 2.0 * opt + 1e-9  # weighted A* is w-admissible
    gr = greedy(g, s, t)
    assert gr.found and gr.cost >= opt - 1e-9
    valid_grid_path(g, gr.path)
    b = bfs(g, s, t)
    assert len(b.path) <= len(dijkstra(g, s, t).path)  # BFS minimises the number of moves


@pytest.mark.parametrize("name", MAPS)
def test_theta_star_any_angle(name):
    g = MAPS[name]
    s, t = endpoints(g)
    th = theta_star(g, s, t)
    assert th.found and th.cost <= dijkstra(g, s, t).cost + 1e-9
    for a, b in zip(th.path[:-1], th.path[1:]):
        assert line_of_sight(g, a, b)  # every straight segment is collision-free


def test_octile_heuristic_is_admissible_and_consistent():
    g = np.zeros((30, 30), bool)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = tuple(rng.integers(0, 30, 2)), tuple(rng.integers(0, 30, 2))
        assert octile(a, b) <= dijkstra(g, a, b).cost + 1e-9
        assert octile(a, b) == pytest.approx(dijkstra(g, a, b).cost)  # exact on an empty grid


def test_unreachable_and_blocked():
    g = np.zeros((20, 20), bool)
    g[:, 10] = True
    assert not astar(g, (5, 2), (5, 15)).found
    assert not astar(g, (5, 10), (5, 15)).found  # start inside an obstacle
    assert not rrt(g, (5.0, 2.0), (5.0, 15.0), iters=500).found


def test_inflation_grows_obstacles():
    g = np.zeros((21, 21), bool)
    g[10, 10] = True
    gi = inflate(g, 3)
    assert gi[10, 13] and gi[12, 12] and not gi[10, 14] and gi.sum() == 29


def test_maze_is_perfect():
    g = maze(cells=6, cell_px=6, seed=2)
    free = np.argwhere(~g)
    # every free cell reachable from the first one (a perfect maze is connected)
    from planning.grid import _neighbors

    seen, stack = {tuple(free[0])}, [tuple(free[0])]
    while stack:
        n = stack.pop()
        for nb, _ in _neighbors(g, n):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == len(free)


def test_rrt_and_rrt_star():
    g = random_obstacles(seed=1)
    s, t = endpoints(g)
    opt = dijkstra(g, s, t).cost
    r = rrt(g, s, t, iters=3000, seed=0)
    rs = rrt(g, s, t, star=True, iters=3000, seed=0)
    assert r.found and rs.found
    for res in (r, rs):
        for a, b in zip(res.path[:-1], res.path[1:]):
            assert line_of_sight(g, a, b)
    assert rs.cost < r.cost                       # same samples, rewiring shortens the path
    assert rs.cost < 1.25 * opt
    # anytime: RRT* cost never increases as samples grow
    costs = [c for _, c in rs.cost_history]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))


def test_rrt_star_tree_costs_are_consistent():
    g = rooms()
    rs = rrt(g, (8.0, 8.0), (90.0, 90.0), star=True, iters=2500, seed=3)
    assert rs.found
    assert rs.cost == pytest.approx(path_length(rs.path), rel=1e-6)


def test_shortcut_shortens_and_stays_valid():
    g = random_obstacles(seed=2)
    s, t = endpoints(g)
    r = rrt(g, s, t, iters=3000, seed=1)
    sc = shortcut(g, r.path, seed=0)
    assert path_length(sc) < path_length(r.path)
    assert sc[0] == tuple(r.path[0]) and sc[-1] == tuple(r.path[-1])
    for a, b in zip(sc[:-1], sc[1:]):
        assert line_of_sight(g, a, b)
