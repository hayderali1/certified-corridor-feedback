"""
plots.py
--------
Publication-quality figures (saved as PDF + PNG) for the paper.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})

C = {"no_filter": "#d1495b", "filter": "#edae49",
     "feedback-basic": "#8d99ae", "feedback-grad": "#2a9d8f",
     "filter+feedback": "#2a9d8f"}
LABELS = {"no_filter": "No filter",
          "filter": "HOCBF filter",
          "feedback-basic": "+ basic feedback",
          "feedback-grad": "+ gradient feedback (ours)",
          "filter+feedback": "+ feedback (ours)"}


def _draw_env(ax, env):
    for (c, r) in env.obstacles:
        ax.add_patch(Circle(c, r, color="#3d3d3d", alpha=0.85, zorder=1))
    ax.plot(*env.start, "o", color="#1d3557", ms=9, zorder=5, label="start")
    ax.plot(*env.goal, "*", color="#e63946", ms=16, zorder=5, label="goal")
    xmin, xmax, ymin, ymax = env.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def trajectory_panels(results_by_config, env, outdir,
                      names=("no_filter", "filter", "feedback-grad")):
    names = [n for n in names if n in results_by_config]
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        res = results_by_config[name]
        _draw_env(ax, env)
        plan = res["plan"]
        px = [p[0] for p in plan]; py = [p[1] for p in plan]
        ax.plot(px, py, "x--", color="#9d4edd", ms=7, mew=2, lw=1.0,
                alpha=0.8, zorder=3, label="LLM waypoints")
        tr = res["trajectory"]
        ax.plot(tr[:, 0], tr[:, 1], "-", color=C[name], lw=2.2, zorder=4,
                label="UAV path")
        outcome = ("COLLISION" if res["collided"]
                   else "REACHED" if res["reached"] else "TIMEOUT")
        tag = LABELS[name]
        if res["replans"]:
            tag += f"\n{res['replans']} re-plans"
        ax.set_title(f"{tag}\noutcome: {outcome}")
        if res["collided"]:
            ax.plot(tr[-1, 0], tr[-1, 1], "X", color="red", ms=14, zorder=6)
        if name == "no_filter":
            ax.legend(loc="lower right", framealpha=0.9)
    fig.suptitle("LLM-guided UAV navigation in a cluttered field "
                 "(identical initial LLM plan across all three)", y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"trajectories.{ext}"))
    plt.close(fig)


def metric_bars(summary, outdir):
    order = ["no_filter", "filter", "feedback-basic", "feedback-grad",
             "filter+feedback"]
    names = [n for n in order if n in summary]
    metrics = [("success_rate", "Success rate (%)"),
               ("collision_rate", "Collision rate (%)"),
               ("timeout_rate", "Timeout rate (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[n][key] for n in names]
        bars = ax.bar([LABELS[n] for n in names], vals,
                      color=[C[n] for n in names], edgecolor="black", lw=0.6)
        ax.set_title(title); ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"metric_bars.{ext}"))
    plt.close(fig)


def halluc_sweep(sweep, outdir):
    order = ["no_filter", "filter", "feedback-basic", "feedback-grad",
             "filter+feedback"]
    names = [n for n in order if n in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        d = sweep[name]
        axes[0].plot(d["halluc"], d["success"], "-o", color=C[name],
                     label=LABELS[name], lw=2)
        axes[1].plot(d["halluc"], d["collision"], "-o", color=C[name],
                     label=LABELS[name], lw=2)
    axes[0].set_title("Task success vs. LLM hallucination rate")
    axes[0].set_xlabel("LLM hallucination rate"); axes[0].set_ylabel("Success rate (%)")
    axes[1].set_title("Collisions vs. LLM hallucination rate")
    axes[1].set_xlabel("LLM hallucination rate"); axes[1].set_ylabel("Collision rate (%)")
    for ax in axes:
        ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"halluc_sweep.{ext}"))
    plt.close(fig)


def safety_timeseries(res, dt, outdir):
    """Min barrier value and intervention magnitude over time (ours)."""
    h = []  # recompute min-barrier proxy already stored? we stored per-step h_min
    interventions = res["interventions"]
    t = np.arange(len(interventions)) * dt
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax1.plot(t, interventions, color="#2a9d8f", lw=1.6,
             label=r"safety-filter intervention $\|u_{safe}-u_{nom}\|$")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("intervention magnitude", color="#2a9d8f")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")
    for j, k in enumerate(res["replan_steps"]):
        ax1.axvline(k * dt, color="#9d4edd", ls="--", lw=1.3,
                    label="re-plan trigger" if j == 0 else None)
    ax1.axhline(0.4, color="grey", ls=":", lw=1, label="re-plan threshold")
    ax1.legend(loc="upper right")
    ax1.set_title("Control-to-language feedback: interventions drive re-planning")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"safety_timeseries.{ext}"))
    plt.close(fig)


def architecture_diagram(outdir):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.04,rounding_size=0.12",
                     fc=fc, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10)

    def arrow(x1, y1, x2, y2, text="", color="black", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=16, lw=1.6,
                     color=color, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text,
                    ha="center", fontsize=8.5, color=color)

    box(0.1, 2.4, 2.0, 1.0, "NL mission\n(operator)", "#e9ecef")
    box(2.6, 2.4, 2.2, 1.0, "LLM planner", "#a8dadc")
    box(5.3, 2.4, 2.2, 1.0, "PD tracking\ncontroller", "#cde7b0")
    box(8.0, 2.4, 2.6, 1.0, "HOCBF-QP\nsafety filter", "#f4a261")
    box(5.3, 0.4, 2.2, 1.0, "UAV dynamics\n(double integrator)", "#dee2e6")
    box(2.6, 0.4, 2.2, 1.0, "deadlock detector\n+ NL feedback", "#ffd6a5")

    arrow(2.1, 2.9, 2.6, 2.9, "task")
    arrow(4.8, 2.9, 5.3, 2.9, "waypoints")
    arrow(7.5, 2.9, 8.0, 2.9, r"$u_{nom}$")
    arrow(9.3, 2.4, 6.5, 1.4, r"$u_{safe}$", color="#bc4749", rad=-0.2)
    arrow(5.3, 0.9, 4.8, 0.9, "state, " + r"$\|u_{safe}-u_{nom}\|$", color="#bc4749")
    arrow(2.6, 1.4, 3.6, 2.4, "re-plan", color="#9d4edd", rad=0.2)
    ax.text(5.5, 3.75, "LLM-guided UAV navigation with HOCBF safety filtering "
            "and control-to-language feedback", ha="center", fontsize=11.5)
    ax.text(3.6, 0.05, "bidirectional control \u2194 language loop (novel)",
            ha="center", fontsize=9, color="#9d4edd")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"architecture.{ext}"))
    plt.close(fig)
