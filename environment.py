"""
environment.py
--------------
Randomized navigation environments. Each environment has a start, a goal,
a field of circular obstacles, and a natural-language mission description
(the kind of instruction an operator would type for the LLM planner).
"""

from __future__ import annotations
import numpy as np


class Environment:
    def __init__(self, start, goal, obstacles, bounds, mission_text):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.obstacles = obstacles            # list of (center (2,), radius)
        self.bounds = bounds                  # (xmin, xmax, ymin, ymax)
        self.mission_text = mission_text

    def obstacle_arrays(self):
        centers = np.array([o[0] for o in self.obstacles])
        radii = np.array([o[1] for o in self.obstacles])
        return centers, radii


def _segment_blocked(a, b, obstacles, clearance=0.0):
    """True if straight segment a->b passes within (r+clearance) of any disk."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    L2 = d @ d
    for (c, r) in obstacles:
        c = np.asarray(c, float)
        t = 0.0 if L2 == 0 else np.clip((c - a) @ d / L2, 0.0, 1.0)
        closest = a + t * d
        if np.linalg.norm(closest - c) < (r + clearance):
            return True
    return False


def generate_environment(rng: np.random.Generator,
                         n_obstacles: int = 6,
                         bounds=(0.0, 10.0, 0.0, 10.0),
                         r_range=(0.6, 1.1),
                         min_clear: float = 0.4):
    """Generate a solvable, non-trivial cluttered environment."""
    xmin, xmax, ymin, ymax = bounds
    start = np.array([xmin + 0.8, ymin + 0.8])
    goal = np.array([xmax - 0.8, ymax - 0.8])

    obstacles = []
    attempts = 0
    while len(obstacles) < n_obstacles and attempts < 2000:
        attempts += 1
        r = rng.uniform(*r_range)
        c = np.array([rng.uniform(xmin + 1.0, xmax - 1.0),
                      rng.uniform(ymin + 1.0, ymax - 1.0)])
        # keep clear of start/goal
        if np.linalg.norm(c - start) < r + 1.0:
            continue
        if np.linalg.norm(c - goal) < r + 1.0:
            continue
        # avoid heavy overlap with existing obstacles
        ok = True
        for (c2, r2) in obstacles:
            if np.linalg.norm(c - c2) < r + r2 + 0.2:
                ok = False
                break
        if ok:
            obstacles.append((c, r))

    mission_text = (
        f"Fly from the launch point at ({start[0]:.1f}, {start[1]:.1f}) to the "
        f"target at ({goal[0]:.1f}, {goal[1]:.1f}). Avoid all obstacles and reach "
        f"the target as directly as possible."
    )
    return Environment(start, goal, obstacles, bounds, mission_text)
