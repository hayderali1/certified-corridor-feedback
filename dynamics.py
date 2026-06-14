"""
dynamics.py
-----------
Guidance-layer UAV model used throughout the framework.

We model the UAV translational dynamics as a double integrator

        p_dot = v
        v_dot = u            (u = commanded acceleration, |u| bounded)

This is the standard guidance/outer-loop abstraction for a multirotor whose
fast inner attitude loop tracks acceleration references (cf. PX4 position/
velocity controllers and the CBF safety-filter literature, e.g. Misyats et
al. 2024). The model is control-affine, which is exactly what the
High-Order Control Barrier Function (HOCBF) safety filter requires.

State  x = [p (dim,), v (dim,)]   ->  shape (2*dim,)
Input  u (dim,)                   ->  commanded acceleration
"""

from __future__ import annotations
import numpy as np


class DoubleIntegratorUAV:
    def __init__(self, dim: int = 2, v_max: float = 2.0, u_max: float = 4.0,
                 dt: float = 0.05):
        self.dim = dim
        self.v_max = float(v_max)     # speed limit  [m/s]
        self.u_max = float(u_max)     # accel limit  [m/s^2] (per-axis box)
        self.dt = float(dt)

    # ----- state accessors -------------------------------------------------
    def pos(self, x: np.ndarray) -> np.ndarray:
        return x[: self.dim]

    def vel(self, x: np.ndarray) -> np.ndarray:
        return x[self.dim:]

    def make_state(self, p, v=None) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(self.dim)
        if v is None:
            v = np.zeros(self.dim)
        v = np.asarray(v, dtype=float).reshape(self.dim)
        return np.concatenate([p, v])

    # ----- continuous dynamics --------------------------------------------
    def f(self, x: np.ndarray) -> np.ndarray:
        """Drift term: [v; 0]."""
        v = self.vel(x)
        return np.concatenate([v, np.zeros(self.dim)])

    def g(self, x: np.ndarray) -> np.ndarray:
        """Control matrix: [0; I]  (shape 2*dim x dim)."""
        I = np.eye(self.dim)
        Z = np.zeros((self.dim, self.dim))
        return np.vstack([Z, I])

    # ----- discrete step (RK4) --------------------------------------------
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float).reshape(self.dim),
                    -self.u_max, self.u_max)

        def xdot(xx):
            return self.f(xx) + self.g(xx) @ u

        dt = self.dt
        k1 = xdot(x)
        k2 = xdot(x + 0.5 * dt * k1)
        k3 = xdot(x + 0.5 * dt * k2)
        k4 = xdot(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # hard speed saturation (physical actuator/airframe limit)
        v = x_next[self.dim:]
        speed = np.linalg.norm(v)
        if speed > self.v_max:
            x_next[self.dim:] = v * (self.v_max / speed)
        return x_next
