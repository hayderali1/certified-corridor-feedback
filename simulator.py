"""
simulator.py
------------
Closed-loop simulation of the full stack:

    NL mission --> LLM planner --> waypoints
                       |
                       v
    PD tracking controller --> u_nom --> HOCBF-QP safety filter --> u_safe
                       ^                                              |
                       |                                              v
            control-to-language  <----- deadlock detector <----- UAV dynamics
                  feedback                (intervention magnitude)

Three configurations are supported via flags:
    use_filter   : engage the HOCBF safety filter (else raw u_nom is applied)
    use_feedback : engage the control-to-language re-planning loop
"""

from __future__ import annotations
import numpy as np
from .dynamics import DoubleIntegratorUAV
from .controller import PDTrackingController
from .cbf_filter import HOCBFSafetyFilter
from . import planner as planner_mod


class SimConfig:
    def __init__(self, use_filter=True, use_feedback=True,
                 feedback_mode="gradient",
                 T_max=30.0, dt=0.05,
                 wp_tol=0.4, goal_tol=0.4,
                 deadlock_window=1.5, deadlock_progress=0.2,
                 deadlock_intervention=0.4, max_replans=5):
        self.use_filter = use_filter
        self.use_feedback = use_feedback
        self.feedback_mode = feedback_mode
        self.T_max = T_max
        self.dt = dt
        self.wp_tol = wp_tol
        self.goal_tol = goal_tol
        self.deadlock_window = deadlock_window            # seconds
        self.deadlock_progress = deadlock_progress        # min progress (m)
        self.deadlock_intervention = deadlock_intervention
        self.max_replans = max_replans


def simulate(env, planner, cfg: SimConfig,
             uav=None, ctrl=None, filt=None):
    if uav is None:
        uav = DoubleIntegratorUAV(dim=2, v_max=2.0, u_max=4.0, dt=cfg.dt)
    if ctrl is None:
        ctrl = PDTrackingController(uav)
    if filt is None:
        filt = HOCBFSafetyFilter(uav)

    x = uav.make_state(env.start, np.zeros(2))
    plan = planner.initial_plan(env)
    active = 0

    n_steps = int(cfg.T_max / cfg.dt)
    win = max(1, int(cfg.deadlock_window / cfg.dt))

    traj = [uav.pos(x).copy()]
    interventions = []
    h_hist = []
    min_clearance = np.inf      # min over time of (||p - c|| - r) to any surface
    dist_window = []          # progress-to-goal tracking
    replans = 0
    replan_steps = []
    collided = False
    reached = False
    infeasible_steps = 0

    for k in range(n_steps):
        p = uav.pos(x)

        # advance through waypoints when close
        while active < len(plan) - 1 and np.linalg.norm(p - plan[active]) < cfg.wp_tol:
            active += 1
        wp = plan[active]

        u_nom = ctrl.command(x, wp)

        if cfg.use_filter:
            res = filt.filter(x, u_nom, env.obstacles)
            u = res["u_safe"]
            interventions.append(res["intervention"])
            h_hist.append(res["h_min"])
            if res["infeasible"]:
                infeasible_steps += 1
            h_argmin = res["h_argmin"]
        else:
            u = u_nom
            interventions.append(0.0)
            # still record true clearance for collision bookkeeping
            hmin = np.inf; h_argmin = -1
            for i, (c, r) in enumerate(env.obstacles):
                hh = np.linalg.norm(p - c) - r
                if hh < hmin:
                    hmin, h_argmin = hh, i
            h_hist.append(hmin)

        # step dynamics
        x = uav.step(x, u)
        p = uav.pos(x)
        traj.append(p.copy())

        # collision check (geometric, against true obstacle radii)
        for (c, r) in env.obstacles:
            clr = np.linalg.norm(p - c) - r
            if clr < min_clearance:
                min_clearance = clr
            if clr <= 0.0:
                collided = True
        if collided:
            break

        # goal check
        if np.linalg.norm(p - env.goal) < cfg.goal_tol:
            reached = True
            break

        # ---- deadlock detection + control-to-language re-planning --------
        dist_window.append(np.linalg.norm(p - env.goal))
        if len(dist_window) > win:
            dist_window.pop(0)
        if cfg.use_filter and cfg.use_feedback and len(dist_window) == win:
            progress = dist_window[0] - dist_window[-1]      # how much closer
            recent_intervention = np.mean(interventions[-win:])
            # (a) low progress while the filter keeps overriding, OR
            # (b) the commanded waypoint itself lies in the unsafe set
            wp_unsafe = any(
                np.linalg.norm(plan[active] - c) <= r + filt.r_safe
                for (c, r) in env.obstacles)
            stalled = ((progress < cfg.deadlock_progress and
                        recent_intervention > cfg.deadlock_intervention)
                       or (wp_unsafe and recent_intervention > cfg.deadlock_intervention))
            if stalled and replans < cfg.max_replans:
                # identify the obstacle the QP fought hardest, build NL feedback
                if 0 <= h_argmin < len(env.obstacles):
                    block = env.obstacles[h_argmin]
                else:
                    block = min(env.obstacles,
                                key=lambda o: np.linalg.norm(p - o[0]))
                if cfg.feedback_mode == "gradient_v2":
                    fb = planner_mod.make_gradient_feedback_v2(
                        block, plan[active], p, env.goal, env.obstacles,
                        filt.r_safe, bounds=env.bounds)
                elif cfg.feedback_mode == "gradient":
                    fb = planner_mod.make_gradient_feedback(
                        block, plan[active], p, env.goal, env.obstacles,
                        filt.r_safe, bounds=env.bounds)
                else:
                    fb = planner_mod.make_feedback(block, plan[active], p)
                plan = planner.replan(env, plan, active, fb)
                active = 0          # new plan is from-here-to-goal; restart it
                replans += 1
                replan_steps.append(k)
                dist_window = []   # reset progress window after re-plan

    traj = np.array(traj)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) \
        if len(traj) > 1 else 0.0
    return {
        "trajectory": traj,
        "reached": bool(reached),
        "collided": bool(collided),
        "timeout": bool(not reached and not collided),
        "path_length": path_len,
        "time": float(len(traj) * cfg.dt),
        "interventions": np.array(interventions),
        "mean_intervention": float(np.mean(interventions)) if interventions else 0.0,
        "min_barrier": float(np.min(h_hist)) if h_hist else np.nan,
        "min_clearance": float(min_clearance),
        "replans": int(replans),
        "replan_steps": list(replan_steps),
        "infeasible_steps": int(infeasible_steps),
        "plan": plan,
        "env": env,
    }
