"""
experiments.py
--------------
Reproducible Monte-Carlo evaluation.

Study A (main): for N randomized environments, compare three configurations
    C0  no_filter        : LLM plan + PD controller, no safety layer
    C1  filter           : + HOCBF-QP safety filter (one-way shield)
    C2  filter+feedback  : + control-to-language feedback re-planning (ours)
All three see the *same* initial plan per environment (planner seeded by the
environment index) so differences are attributable to the architecture only.

Study B (sweep): vary the LLM hallucination rate and measure how success and
collision rates respond for each configuration -> robustness characterisation.
"""

from __future__ import annotations
import numpy as np
import csv
import os
from .environment import generate_environment
from .planner import SurrogateLLMPlanner
from .simulator import simulate, SimConfig

CONFIGS = [
    ("no_filter", dict(use_filter=False, use_feedback=False)),
    ("filter", dict(use_filter=True, use_feedback=False)),
    ("feedback-basic", dict(use_filter=True, use_feedback=True,
                            feedback_mode="basic")),
    ("feedback-grad", dict(use_filter=True, use_feedback=True,
                           feedback_mode="gradient")),
    ("feedback-grad-v2", dict(use_filter=True, use_feedback=True,
                              feedback_mode="gradient_v2")),
]


def run_main_study(n_envs=120, n_obstacles=6, halluc_rate=0.6, seed0=1000):
    rows = []
    per_config = {name: [] for name, _ in CONFIGS}
    for i in range(n_envs):
        env = generate_environment(np.random.default_rng(seed0 + i),
                                   n_obstacles=n_obstacles)
        for name, flags in CONFIGS:
            # identical initial plan across configs (planner seeded per env)
            planner = SurrogateLLMPlanner(np.random.default_rng(seed0 + i),
                                          halluc_rate=halluc_rate)
            res = simulate(env, planner, SimConfig(**flags))
            rec = dict(env=i, config=name,
                       reached=int(res["reached"]),
                       collided=int(res["collided"]),
                       timeout=int(res["timeout"]),
                       path_length=res["path_length"],
                       time=res["time"],
                       replans=res["replans"],
                       mean_intervention=res["mean_intervention"],
                       min_barrier=res["min_barrier"],
                       min_clearance=res["min_clearance"],
                       infeasible_steps=res["infeasible_steps"])
            rows.append(rec)
            per_config[name].append(res)
    return rows, per_config


def summarize(rows):
    """Aggregate metrics per configuration."""
    summary = {}
    for name, _ in CONFIGS:
        r = [x for x in rows if x["config"] == name]
        n = len(r)
        succ = [x for x in r if x["reached"]]
        summary[name] = dict(
            n=n,
            success_rate=100.0 * sum(x["reached"] for x in r) / n,
            collision_rate=100.0 * sum(x["collided"] for x in r) / n,
            timeout_rate=100.0 * sum(x["timeout"] for x in r) / n,
            mean_path_len=(np.mean([x["path_length"] for x in succ])
                           if succ else float("nan")),
            mean_time=(np.mean([x["time"] for x in succ])
                       if succ else float("nan")),
            mean_replans=np.mean([x["replans"] for x in r]),
            min_barrier=np.min([x["min_barrier"] for x in r]),
            min_clearance=np.min([x["min_clearance"] for x in r]),
        )
    return summary


def run_halluc_sweep(levels=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     n_envs=50, n_obstacles=6, seed0=5000):
    out = {name: {"halluc": [], "success": [], "collision": []}
           for name, _ in CONFIGS}
    for h in levels:
        rows = []
        for i in range(n_envs):
            env = generate_environment(np.random.default_rng(seed0 + i),
                                       n_obstacles=n_obstacles)
            for name, flags in CONFIGS:
                planner = SurrogateLLMPlanner(np.random.default_rng(seed0 + i),
                                              halluc_rate=h)
                res = simulate(env, planner, SimConfig(**flags))
                rows.append((name, res["reached"], res["collided"]))
        for name, _ in CONFIGS:
            r = [x for x in rows if x[0] == name]
            out[name]["halluc"].append(h)
            out[name]["success"].append(100.0 * sum(x[1] for x in r) / len(r))
            out[name]["collision"].append(100.0 * sum(x[2] for x in r) / len(r))
    return out


def save_rows_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
