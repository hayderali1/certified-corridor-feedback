"""
geometry.py
-----------
Control-derived spatial reasoning used to enrich the feedback to the planner.

When the safety filter deadlocks the UAV against an obstacle, the *barrier
gradient* of the tightest constraint already encodes local safe-direction
information. This module turns that geometry into (i) an open "escape" bearing
and (ii) a small set of control-validated detour waypoints that hug the blocking
obstacle on each side and are checked for clearance against *all* obstacles.

These quantities are what the gradient-informed feedback verbalizes for the LLM,
so the planner re-plans with control-grounded spatial cues instead of guessing.
"""

from __future__ import annotations
import numpy as np

_COMPASS = ["east", "north-east", "north", "north-west",
            "west", "south-west", "south", "south-east"]


def _rot(u, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def bearing_text(direction):
    """8-wind compass label for a 2-D direction vector."""
    ang = np.arctan2(direction[1], direction[0])           # [-pi, pi]
    idx = int(np.round(ang / (np.pi / 4))) % 8
    return _COMPASS[idx]


def bypass_geometry(p, goal, blocking, obstacles, r_safe,
                    margin=0.6, bounds=None):
    """
    Compute control-validated detour candidates around `blocking`.

    The barrier gradient (outward radial) and the obstacle's tangent cone define
    the local *open* region. We sample waypoints on rings around the blocking
    obstacle and score each by (i) feasibility: clearance to ALL obstacles, and
    (ii) progress: how much closer it is to the goal than the stalled pose. The
    top-ranked feasible, goal-advancing candidates are returned. This keeps the
    cue control-grounded while remaining robust in dense clutter.

    Returns dict: candidates (ranked), escape_dir, bearing, grad.
    """
    p = np.asarray(p, float)
    goal = np.asarray(goal, float)
    c, r = np.asarray(blocking[0], float), float(blocking[1])
    R = r + r_safe
    to_c = c - p
    d = float(np.linalg.norm(to_c))
    grad = (p - c) / d if d > 1e-9 else np.array([1.0, 0.0])   # outward radial
    d_goal_now = float(np.linalg.norm(p - goal))

    base_ang = np.arctan2(p[1] - c[1], p[0] - c[0])   # UAV bearing about centre
    cand = []
    for m in (margin, margin + 0.6, margin + 1.2):
        ring = R + m
        for k in range(-4, 5):                        # arc around the obstacle
            ang = base_ang + k * (np.pi / 7.0)
            cand.append(c + ring * np.array([np.cos(ang), np.sin(ang)]))

    def clearance(wp):
        return min(np.linalg.norm(wp - np.asarray(oc, float)) - orr
                   for (oc, orr) in obstacles)

    def in_bounds(wp):
        if bounds is None:
            return True
        xmin, xmax, ymin, ymax = bounds
        return xmin - 0.3 <= wp[0] <= xmax + 0.3 and ymin - 0.3 <= wp[1] <= ymax + 0.3

    def score(wp):
        clr = clearance(wp)
        progress = d_goal_now - float(np.linalg.norm(wp - goal))
        feasible = 1 if (clr > r_safe and in_bounds(wp)) else 0
        # prefer feasible + goal-advancing; among those, more clearance
        return (feasible, progress > 0.0, progress + 0.5 * clr)

    cand.sort(key=score, reverse=True)
    cand = cand[:4]
    escape_dir = cand[0] - p
    nrm = np.linalg.norm(escape_dir)
    escape_dir = escape_dir / nrm if nrm > 1e-9 else grad
    return {
        "candidates": cand,
        "escape_dir": escape_dir,
        "bearing": bearing_text(escape_dir),
        "grad": grad,
    }
