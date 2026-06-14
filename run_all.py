"""
run_all.py
----------
Reproduce every figure and table in the paper.

    python run_all.py            # full study (default sizes)
    python run_all.py --quick    # smaller sizes for a fast smoke run
"""

from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from uav_llm_cbf import generate_environment, SurrogateLLMPlanner, simulate, SimConfig
from uav_llm_cbf import experiments as ex
from uav_llm_cbf import plots

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results", "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)


def example_run(seed=7, halluc=0.6):
    env = generate_environment(np.random.default_rng(seed), n_obstacles=6)
    out = {}
    for name, flags in ex.CONFIGS:
        planner = SurrogateLLMPlanner(np.random.default_rng(123), halluc_rate=halluc)
        out[name] = simulate(env, planner, SimConfig(**flags))
    return env, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_main", type=int, default=None)
    ap.add_argument("--n_sweep", type=int, default=None)
    args = ap.parse_args()

    n_main = 40 if args.quick else 120
    n_sweep = 20 if args.quick else 50
    if args.n_main is not None:
        n_main = args.n_main
    if args.n_sweep is not None:
        n_sweep = args.n_sweep
    t0 = time.time()

    print("[1/5] architecture diagram ...")
    plots.architecture_diagram(FIG)

    print("[2/5] example trajectories (seed 7) ...")
    env, ex_out = example_run()
    plots.trajectory_panels(ex_out, env, FIG)
    plots.safety_timeseries(ex_out["feedback-grad"], dt=0.05, outdir=FIG)

    print(f"[3/5] main study (n_envs={n_main}) ...")
    rows, _ = ex.run_main_study(n_envs=n_main)
    ex.save_rows_csv(rows, os.path.join(RES, "main_study.csv"))
    summary = ex.summarize(rows)
    plots.metric_bars(summary, FIG)

    print(f"[4/5] hallucination sweep (n_envs={n_sweep}/level) ...")
    sweep = ex.run_halluc_sweep(n_envs=n_sweep)
    plots.halluc_sweep(sweep, FIG)

    print("[5/5] writing summary ...")
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump({"summary": summary, "sweep": sweep}, f, indent=2)

    # pretty table
    hdr = (f"{'config':18s} {'success%':>9s} {'collision%':>11s} "
           f"{'timeout%':>9s} {'pathlen':>8s} {'time(s)':>8s} "
           f"{'replans':>8s} {'min_clear(m)':>13s}")
    lines = [hdr, "-" * len(hdr)]
    for name, _ in ex.CONFIGS:
        s = summary[name]
        lines.append(f"{name:18s} {s['success_rate']:9.1f} "
                     f"{s['collision_rate']:11.1f} {s['timeout_rate']:9.1f} "
                     f"{s['mean_path_len']:8.2f} {s['mean_time']:8.2f} "
                     f"{s['mean_replans']:8.2f} {s['min_clearance']:13.3f}")
    table = "\n".join(lines)
    with open(os.path.join(RES, "summary_table.txt"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nDone in {time.time()-t0:.1f}s. Figures -> {FIG}")


if __name__ == "__main__":
    main()
