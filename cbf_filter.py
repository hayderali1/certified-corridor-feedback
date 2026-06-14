"""
cbf_filter.py
-------------
High-Order Control Barrier Function (HOCBF) Quadratic-Program safety filter.

This is the formal safety layer of the framework. It takes the *nominal*
acceleration command u_nom produced by the tracking controller (which is
driving the UAV toward whatever waypoint the LLM proposed) and returns the
*minimally modified* safe command u_safe that is guaranteed to keep the UAV
inside the safe set (collision-free + speed-limited).

Obstacle i is a disk of radius r_i centred at p_o_i. Define the barrier

        h_i(p) = ||p - p_o_i||^2 - (r_i + r_safe)^2 .

For the double-integrator UAV, h_i has relative degree 2 w.r.t. u, so we use
an exponential / High-Order CBF with linear class-K functions a1, a2:

        psi0 = h_i
        psi1 = h_i_dot + a1 * h_i
        constraint:  psi1_dot + a2 * psi1 >= 0 .

Expanding (p_dot = v, v_dot = u) gives a constraint that is *linear in u*:

   2 (p-p_o)·u  >=  -( 2||v||^2 + (a1+a2) 2(p-p_o)·v + a1 a2 h_i ) .

A speed (velocity) CBF h_v = v_max^2 - ||v||^2 (relative degree 1) adds

   2 v·u  <=  a_v ( v_max^2 - ||v||^2 ) .

The filter solves the QP

   min_u || u - u_nom ||^2
   s.t.   HOCBF (per obstacle), speed CBF, box |u| <= u_max .

The optimal value's deviation ||u_safe - u_nom|| is the *intervention
magnitude*: a control-theoretic scalar that we later convert into natural-
language feedback for the LLM (the key novelty of the framework).
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp


class HOCBFSafetyFilter:
    def __init__(self, uav, r_safe: float = 0.25,
                 a1: float = 3.0, a2: float = 3.0, a_v: float = 4.0,
                 max_obstacles: int = 16):
        self.uav = uav
        self.dim = uav.dim
        self.r_safe = float(r_safe)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a_v = float(a_v)
        self.max_obs = int(max_obstacles)

        d = self.dim
        # ---- parametrized QP (built once, re-solved fast with OSQP) -------
        self.u = cp.Variable(d)
        self.u_nom = cp.Parameter(d)
        self.A_obs = cp.Parameter((self.max_obs, d))   # rows: 2(p-p_o)
        self.b_obs = cp.Parameter(self.max_obs)        # rows: rhs lower bnd
        self.v_row = cp.Parameter(d)                   # 2 v
        self.v_rhs = cp.Parameter()                    # a_v(v_max^2-||v||^2)

        cons = [
            self.A_obs @ self.u >= self.b_obs,         # obstacle HOCBFs
            self.v_row @ self.u <= self.v_rhs,         # speed CBF
            self.u <= uav.u_max,
            self.u >= -uav.u_max,
        ]
        obj = cp.Minimize(cp.sum_squares(self.u - self.u_nom))
        self.prob = cp.Problem(obj, cons)

    # ----------------------------------------------------------------------
    def _barriers(self, x, obstacles):
        """Return list of (h_i, p_o_i, R_i, dist_surface_i) for all obstacles."""
        p = self.uav.pos(x)
        out = []
        for (po, r) in obstacles:
            po = np.asarray(po, float)
            R = r + self.r_safe
            diff = p - po
            h = float(diff @ diff - R * R)
            out.append((h, po, R, float(np.linalg.norm(diff) - R)))
        return out

    def filter(self, x, u_nom, obstacles):
        """
        Return dict with safe command and diagnostics.

        obstacles : list of (center (dim,), radius float)
        """
        d = self.dim
        p = self.uav.pos(x)
        v = self.uav.vel(x)
        u_nom = np.asarray(u_nom, float).reshape(d)

        # build obstacle HOCBF rows
        A = np.zeros((self.max_obs, d))
        b = np.full(self.max_obs, -1e6)   # disabled slots: 0·u >= -1e6 (trivial)
        bar = self._barriers(x, obstacles)
        n = min(len(bar), self.max_obs)
        h_min = np.inf
        h_argmin = -1
        for i in range(n):
            h, po, R, _ = bar[i]
            diff = p - po
            A[i] = 2.0 * diff
            hdot = 2.0 * diff @ v
            b[i] = -(2.0 * (v @ v)
                     + (self.a1 + self.a2) * hdot
                     + self.a1 * self.a2 * h)
            if h < h_min:
                h_min, h_argmin = h, i

        # speed CBF
        v_row = 2.0 * v
        v_rhs = self.a_v * (self.uav.v_max ** 2 - v @ v)

        # set parameters
        self.A_obs.value = A
        self.b_obs.value = b
        self.v_row.value = v_row
        self.v_rhs.value = v_rhs
        self.u_nom.value = u_nom

        infeasible = False
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=False,
                            max_iter=10000, eps_abs=1e-6, eps_rel=1e-6,
                            verbose=False)
            if self.u.value is None or self.prob.status not in (
                    "optimal", "optimal_inaccurate"):
                infeasible = True
        except Exception:
            infeasible = True

        if infeasible:
            # safety fallback: maximal admissible braking
            u_safe = np.clip(-3.0 * v, -self.uav.u_max, self.uav.u_max)
        else:
            u_safe = np.clip(self.u.value, -self.uav.u_max, self.uav.u_max)

        intervention = float(np.linalg.norm(u_safe - u_nom))
        return {
            "u_safe": u_safe,
            "intervention": intervention,     # ||u_safe - u_nom||
            "h_min": float(h_min),            # tightest barrier value
            "h_argmin": int(h_argmin),        # index of tightest obstacle
            "infeasible": bool(infeasible),
            "barriers": bar,
        }
