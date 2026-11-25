"""Benchmark + figures + GIFs:  python -m planning.viz  ->  docs/"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from . import astar, bfs, dijkstra, greedy, maze, random_obstacles, rooms, rrt, shortcut, theta_star, weighted_astar

C = {"path": "#e63946", "explored": "#a8dadc", "tree": "#457b9d", "start": "#2a9d8f", "goal": "#f4a261"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _base(ax, grid, title):
    ax.imshow(grid, cmap="Greys", origin="lower", interpolation="nearest", vmin=0, vmax=1.3)
    ax.set_title(title, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _endpoints(ax, s, g):
    ax.plot(s[1], s[0], "o", ms=9, color=C["start"], zorder=6)
    ax.plot(g[1], g[0], "*", ms=14, color=C["goal"], mec="black", mew=0.6, zorder=6)


def _draw_path(ax, path, color=C["path"], lw=2.4):
    if path:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], color=color, lw=lw, zorder=5)


def _quantize(path: Path, colors: int = 96) -> None:
    from PIL import Image

    Image.open(path).convert("RGB").quantize(colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def fig_compare(out: Path):
    plt = _plt()
    g = random_obstacles(seed=1)
    H, W = g.shape
    s, t = (3, 3), (H - 4, W - 4)
    runs = [bfs, dijkstra, astar, lambda a, b, c, **k: weighted_astar(a, b, c, 2.0, **k), greedy, theta_star]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for ax, f in zip(axes.ravel()[:6], runs):
        r = f(g, s, t, record=True)
        _base(ax, g, r.name)
        if r.order:
            o = np.array(r.order)
            ax.scatter(o[:, 1], o[:, 0], s=3, color=C["explored"], lw=0, zorder=2)
        _draw_path(ax, r.path)
        _endpoints(ax, s, t)
        ax.text(0.02, 0.02, f"length {r.cost:.1f} · expanded {r.expanded:,}", transform=ax.transAxes,
                fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85), zorder=7)
    for ax, star in zip(axes.ravel()[6:], (False, True)):
        r = rrt(g, s, t, star=star, iters=3000, seed=0)
        _base(ax, g, r.name + " (3,000 samples)")
        P = r.points
        for a, b in r.edges:
            ax.plot([P[a, 1], P[b, 1]], [P[a, 0], P[b, 0]], color=C["tree"], lw=0.5, alpha=0.6, zorder=2)
        _draw_path(ax, r.path)
        _endpoints(ax, s, t)
        ax.text(0.02, 0.02, f"length {r.cost:.1f} · nodes {r.nodes:,}", transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.85), zorder=7)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(out / "planners.png", dpi=95, facecolor="white")
    plt.close(fig)
    _quantize(out / "planners.png")


def fig_maze(out: Path):
    plt = _plt()
    g = maze(seed=3)
    H, W = g.shape
    s, t = (4, 4), (H - 5, W - 5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    for ax, f in zip(axes, (dijkstra, astar, greedy)):
        r = f(g, s, t, record=True)
        _base(ax, g, r.name)
        o = np.array(r.order)
        ax.scatter(o[:, 1], o[:, 0], s=4, c=np.arange(len(o)), cmap="viridis", lw=0, zorder=2, rasterized=True)
        _draw_path(ax, r.path)
        _endpoints(ax, s, t)
        ax.text(0.02, -0.07, f"length {r.cost:.1f} · expanded {r.expanded:,}", transform=ax.transAxes, fontsize=9)
    fig.suptitle("In a maze the heuristic barely helps: walls fool it (colour = expansion order)", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "maze.png", dpi=80, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _quantize(out / "maze.png")


def gif_astar(out: Path):
    plt = _plt()
    from PIL import Image

    g = rooms()
    H, W = g.shape
    s, t = (8, 8), (H - 9, W - 9)
    r = astar(g, s, t, record=True)
    order = np.array(r.order)
    frames = []
    cuts = list(np.linspace(1, len(order), 40).astype(int)) + [len(order)] * 12
    for k, n in enumerate(cuts):
        fig, ax = plt.subplots(figsize=(5, 5))
        _base(ax, g, f"A* search · {n:,} cells expanded")
        o = order[:n]
        ax.scatter(o[:, 1], o[:, 0], s=6, color=C["explored"], lw=0, zorder=2)
        ax.scatter(o[-40:, 1], o[-40:, 0], s=8, color=C["tree"], lw=0, zorder=3)
        if k >= 40:
            _draw_path(ax, r.path)
        _endpoints(ax, s, t)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=70, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=48))
    frames[0].save(out / "astar.gif", save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


def gif_rrtstar(out: Path):
    plt = _plt()
    from PIL import Image

    g = random_obstacles(seed=4)
    H, W = g.shape
    s, t = (3, 3), (H - 4, W - 4)
    frames = []
    for iters in list(range(150, 1500, 150)) + list(range(1500, 6001, 500)) + [6000] * 6:
        r = rrt(g, s, t, star=True, iters=iters, seed=2)
        fig, ax = plt.subplots(figsize=(5, 5))
        _base(ax, g, f"RRT* · {iters:,} samples" + (f" · length {r.cost:.1f}" if r.found else ""))
        P = r.points
        for a, b in r.edges:
            ax.plot([P[a, 1], P[b, 1]], [P[a, 0], P[b, 0]], color=C["tree"], lw=0.5, alpha=0.55)
        _draw_path(ax, r.path)
        _endpoints(ax, s, t)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=70, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=48))
    frames[0].save(out / "rrtstar.gif", save_all=True, append_images=frames[1:], duration=260, loop=0, optimize=True)


def benchmark(out: Path, n_maps: int = 20) -> dict:
    plt = _plt()
    planners = {"Dijkstra": dijkstra, "A*": astar, "Weighted A* (w=2)": lambda a, b, c: weighted_astar(a, b, c, 2.0),
                "Greedy": greedy, "Theta*": theta_star}
    rows = {k: {"ratio": [], "expanded": [], "ms": [], "found": 0} for k in list(planners) + ["RRT", "RRT* ", "RRT + shortcut"]}
    used = 0
    for seed in range(100):
        if used >= n_maps:
            break
        g = random_obstacles(seed=seed)
        H, W = g.shape
        s, t = (3, 3), (H - 4, W - 4)
        opt = dijkstra(g, s, t)
        if not opt.found:
            continue
        used += 1
        for name, f in planners.items():
            r = f(g, s, t)
            rows[name]["found"] += r.found
            if r.found:
                rows[name]["ratio"].append(r.cost / opt.cost)
                rows[name]["expanded"].append(r.expanded)
                rows[name]["ms"].append(r.seconds * 1000)
        for name, star in (("RRT", False), ("RRT* ", True)):
            r = rrt(g, s, t, star=star, iters=3000, seed=seed)
            rows[name]["found"] += r.found
            if r.found:
                rows[name]["ratio"].append(r.cost / opt.cost)
                rows[name]["expanded"].append(r.nodes)
                rows[name]["ms"].append(r.seconds * 1000)
                if not star:
                    sc = shortcut(g, r.path, seed=seed)
                    from .grid import path_length

                    rows["RRT + shortcut"]["found"] += 1
                    rows["RRT + shortcut"]["ratio"].append(path_length(sc) / opt.cost)
                    rows["RRT + shortcut"]["expanded"].append(r.nodes)
                    rows["RRT + shortcut"]["ms"].append(r.seconds * 1000)
    summary = {k: {"success": f"{v['found']}/{used}",
                   "length_vs_optimal": round(float(np.mean(v["ratio"])), 3) if v["ratio"] else None,
                   "expanded_or_nodes": int(np.median(v["expanded"])) if v["expanded"] else None,
                   "ms": round(float(np.median(v["ms"])), 1) if v["ms"] else None} for k, v in rows.items()}
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2))

    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ratios = [summary[k]["length_vs_optimal"] or math.nan for k in names]
    axes[0].barh(names, [100 * (r - 1) for r in ratios], color=["#457b9d"] * 5 + ["#e76f51"] * 3)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set(title=f"Path length vs 8-connected optimum ({used} random maps)", xlabel="% longer (negative = shorter)")
    axes[0].invert_yaxis()
    ms = [summary[k]["ms"] or math.nan for k in names]
    axes[1].barh(names, ms, color=["#457b9d"] * 5 + ["#e76f51"] * 3)
    axes[1].set(title="Median planning time", xlabel="ms (Python)")
    axes[1].set_xscale("log")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=105, facecolor="white")
    plt.close(fig)
    return summary


def main() -> int:
    out = Path("docs")
    out.mkdir(exist_ok=True)
    fig_compare(out)
    fig_maze(out)
    gif_astar(out)
    gif_rrtstar(out)
    print(json.dumps(benchmark(out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
