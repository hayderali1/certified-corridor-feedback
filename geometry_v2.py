"""
geometry_v2.py
--------------
Improved control-derived spatial feedback (fixes + extension of geometry.py).

Fixes over v1
-------------
1. FULL-CIRCLE candidate sampling. v1 sampled an arc of only +-4*(pi/7)
   (~ +-103 deg) around the UAV's approach bearing, so escapes "behind" the
   blocking obstacle were invisible.
2. LINE-OF-SIGHT (segment) validation. v1 scored candidates by *point*
   clearance only; a candidate could be clear yet unreachable because the
   straight leg stall->candidate (or candidate->goal) crosses another
   obstacle. In 10-obstacle clutter this is the dominant failure: the UAV
   detours into the next deadlock and burns the re-plan budget.
3. MULTI-HOP corridor. When no single detour has line-of-sight to the goal,
   v2 runs Dijkstra over a small visibility graph of ring points around ALL
   obstacles (inflated by r_safe + margin) and returns the first hops of the
   shortest safe path. This is exactly the "multi-step gradient feedback"
   the paper lists as future work, still 100% control-derived.

The returned dict is backward compatible (candidates / escape_dir / bearing /
grad) and additionally carries `path` (ordered corridor waypoints) and
`los_goal` (whether the last hop sees the goal).
"""

from __future__ import annotations
import heapq
import numpy as np

_COMPASS = ["east", "north-east", "north", "north-west",
            "west", "south-west", "south", "south-east"]


def bearing_text(direction):
    ang = np.arctan2(direction[1], direction[0])
    idx = int(np.round(ang / (np.pi / 4))) % 8
    return _COMPASS[idx]


def _seg_clear(a, b, obstacles, clearance):
    """True if straight segment a->b keeps `clearance` to every disk."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    L2 = float(d @ d)
    for (c, r) in obstacles:
        c = np.asarray(c, float)
        t = 0.0 if L2 == 0.0 else float(np.clip((c - a) @ d / L2, 0.0, 1.0))
        if np.linalg.norm(a + t * d - c) < (r + clearance):
            return False
    return True


def bypass_geometry_v2(p, goal, blocking, obstacles, r_safe,
                       margin=0.35, n_ring=12, bounds=None, max_hops=3):
    """
    Control-validated detour corridor around the blocking obstacle.

    Returns dict with:
        path       : list of corridor waypoints (1..max_hops), each leg
                     line-of-sight clear of all inflated obstacles
        candidates : path (alias, backward compatible)
        escape_dir, bearing, grad, los_goal
    """
    p = np.asarray(p, float)
    goal = np.asarray(goal, float)
    c_blk = np.asarray(blocking[0], float)
    d_blk = float(np.linalg.norm(p - c_blk))
    grad = (p - c_blk) / d_blk if d_blk > 1e-9 else np.array([1.0, 0.0])

    clr_req = r_safe + 0.05          # leg clearance the corridor must keep
    ring_off = r_safe + margin       # how far ring nodes sit off the surface

    def in_bounds(q):
        if bounds is None:
            return True
        xmin, xmax, ymin, ymax = bounds
        return (xmin - 0.3 <= q[0] <= xmax + 0.3 and
                ymin - 0.3 <= q[1] <= ymax + 0.3)

    def point_clear(q):
        return min(np.linalg.norm(q - np.asarray(oc, float)) - orr
                   for (oc, orr) in obstacles) > clr_req - 1e-9

    # --- nodes: full ring around EVERY obstacle (not just the blocker) ----
    nodes = [p, goal]                                   # 0 = start, 1 = goal
    for (oc, orr) in obstacles:
        oc = np.asarray(oc, float)
        for k in range(n_ring):
            a = 2.0 * np.pi * k / n_ring
            q = oc + (orr + ring_off) * np.array([np.cos(a), np.sin(a)])
            if in_bounds(q) and point_clear(q):
                nodes.append(q)

    n = len(nodes)
    # --- Dijkstra over the visibility graph ------------------------------
    dist = [np.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        if i == 1:
            break
        for j in range(n):
            if j == i:
                continue
            nd = d + float(np.linalg.norm(nodes[i] - nodes[j]))
            if nd >= dist[j]:
                continue                       # cheap reject before LOS test
            if _seg_clear(nodes[i], nodes[j], obstacles, clr_req):
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))

    # --- extract corridor --------------------------------------------------
    if np.isfinite(dist[1]):
        path_idx = []
        j = 1
        while j != -1:
            path_idx.append(j)
            j = prev[j]
        path_idx.reverse()                      # start ... goal
        hops = [nodes[j] for j in path_idx[1:-1]][:max_hops]
        los_goal = len(path_idx) - 2 <= max_hops
        if not hops:                            # direct LOS to goal
            hops = [goal.copy()]
    else:
        # graph disconnected (extreme clutter): fall back to best ring point
        # of the blocking obstacle by point clearance + goal progress.
        best, best_s = None, -np.inf
        for k in range(n_ring):
            a = 2.0 * np.pi * k / n_ring
            q = c_blk + (blocking[1] + ring_off) * np.array([np.cos(a), np.sin(a)])
            if not (in_bounds(q) and point_clear(q)):
                continue
            s = (np.linalg.norm(p - goal) - np.linalg.norm(q - goal))
            if s > best_s:
                best, best_s = q, s
        hops = [best if best is not None else p + grad * (r_safe + margin)]
        los_goal = False

    esc = hops[0] - p
    nrm = float(np.linalg.norm(esc))
    esc = esc / nrm if nrm > 1e-9 else grad
    return {
        "path": hops,
        "candidates": hops,          # backward-compatible alias
        "escape_dir": esc,
        "bearing": bearing_text(esc),
        "grad": grad,
        "los_goal": bool(los_goal),
    }
