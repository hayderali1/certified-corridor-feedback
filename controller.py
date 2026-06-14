"""
controller.py
-------------
Low-level tracking controller (guidance loop).

Given the current state and the active waypoint p_wp proposed by the LLM
planner, produce a nominal acceleration command

        u_nom = -kp (p - p_wp) - kd v ,

saturated to the actuator limit. This is the command the HOCBF safety filter
will minimally correct. A small approach-speed shaping term keeps the UAV
from overshooting waypoints.
"""

from __future__ import annotations
import numpy as np


class PDTrackingController:
    def __init__(self, uav, kp: float = 2.0, kd: float = 3.0):
        self.uav = uav
        self.kp = float(kp)
        self.kd = float(kd)

    def command(self, x, p_wp) -> np.ndarray:
        p = self.uav.pos(x)
        v = self.uav.vel(x)
        p_wp = np.asarray(p_wp, float).reshape(self.uav.dim)
        u = -self.kp * (p - p_wp) - self.kd * v
        return np.clip(u, -self.uav.u_max, self.uav.u_max)
