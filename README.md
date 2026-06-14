# Safe LLM-Guided UAV Navigation via HOCBF Safety Filtering and Control-to-Language Feedback

Reproducible simulation framework for an ASYU 2026 submission. An LLM proposes
waypoints from a natural-language mission; a PD controller tracks them; a
**High-Order Control Barrier Function (HOCBF) quadratic program** minimally
corrects the command to guarantee collision avoidance and speed/actuator limits;
and a **control-to-language feedback loop** converts the filter's intervention
into natural-language feedback that triggers LLM re-planning.

## Novel contribution: gradient-informed feedback
The core novelty is *what* the controller says back to the planner. When the
filter deadlocks the UAV against an obstacle, we read the **barrier gradient**
(the direction the filter is already pushing) and the obstacle tangent cone,
sample that open region, validate candidates against *all* obstacles and goal
progress, and verbalize an **open bearing plus control-validated detour
waypoints** for the LLM. The planner then re-plans with control-grounded spatial
cues instead of guessing. The framework ships a `feedback_mode` flag so this is a
built-in ablation:

| Config | Feedback | What it isolates |
|---|---|---|
| `no_filter` | – | raw LLM danger |
| `filter` | – | one-way shield (prior art) |
| `feedback-basic` | "you're stuck, detour" | re-prompting alone |
| `feedback-grad` (ours) | gradient bearing + validated waypoints | the novel control→language cue |

On the surrogate at 10 obstacles (n=60), the ablation shows the cue matters:
filter 0% → basic feedback 41.7% → **gradient feedback 58.3%** success, with
*fewer* re-plans (3.67 vs 4.25) and 0% collisions throughout. The same flag works
with any LLM backend, so you can reproduce this gap on real models.

## Install
```bash
cd uav_llm_cbf
python3 -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

## Run everything (figures + tables)
From the directory that CONTAINS the `uav_llm_cbf` folder (i.e. its parent):
```bash
python3 -m uav_llm_cbf.run_all                 # full study (slow, ~10-15 min)
python3 -m uav_llm_cbf.run_all --quick         # fast smoke run (~5 min)
python3 -m uav_llm_cbf.run_all --n_main 80 --n_sweep 20   # custom sizes
```
Outputs are written to `uav_llm_cbf/results/`:
- `figures/architecture.{pdf,png}` — system diagram
- `figures/trajectories.{pdf,png}` — 3-config comparison on one environment
- `figures/metric_bars.{pdf,png}` — success/collision/timeout rates
- `figures/halluc_sweep.{pdf,png}` — robustness vs. LLM hallucination rate
- `figures/safety_timeseries.{pdf,png}` — interventions + re-plan triggers
- `main_study.csv`, `summary.json`, `summary_table.txt`

The pre-generated figures and results from our run are already included.

## Quick single-environment demo
```python
import numpy as np
from uav_llm_cbf import generate_environment, SurrogateLLMPlanner, simulate, SimConfig

env = generate_environment(np.random.default_rng(7), n_obstacles=6)
planner = SurrogateLLMPlanner(np.random.default_rng(123), halluc_rate=0.6)
res = simulate(env, planner, SimConfig(use_filter=True, use_feedback=True))
print("reached:", res["reached"], "collided:", res["collided"],
      "re-plans:", res["replans"], "min clearance (m):", round(res["min_clearance"], 3))
```

## Headline results (N=80 environments, hallucination rate 0.6)
| Config | Success | Collision | Min clearance | Re-plans |
|---|---|---|---|---|
| No filter | 0.0% | 100.0% | -0.05 m | 0 |
| HOCBF filter (shield) | 3.8% | **0.0%** | +0.25 m | 0 |
| HOCBF + feedback (ours) | **71.2%** | **0.0%** | +0.25 m | 3.3 |

Across hallucination rates 0→1: both filtered configs stay at **0% collisions**;
ours sustains **65–90% success** while the shield collapses to 0%.

## Using a real LLM (camera-ready)
The default planner is a calibrated surrogate so the Monte-Carlo study is
reproducible and free. To run with a real Claude model, set your key and swap the
backend:
```bash
export ANTHROPIC_API_KEY=sk-...
pip install anthropic
```
```python
from uav_llm_cbf.planner import GroqLLMPlanner
planner = GroqLLMPlanner(model="llama-3.3-70b-versatile")   # same interface
```
`AnthropicLLMPlanner` / `GroqLLMPlanner` implement `initial_plan` / `replan`
exactly like the surrogate, so `simulate(...)` and `experiments.py` work
unchanged. The experiment set now runs **4 configurations** (adding
`feedback-basic` vs `feedback-grad`), so `run_llm_groq.py` issues extra re-plan
calls for both feedback variants — start with a small `--n_envs` to gauge usage.
To compare LLMs, just change `--model`; the gradient-feedback cue is identical
across models, so any success gap reflects the model, not the harness.

## Code layout
```
uav_llm_cbf/
  dynamics.py      double-integrator UAV (control-affine guidance model)
  controller.py    PD tracking controller -> nominal acceleration
  cbf_filter.py    HOCBF-QP safety filter (OSQP via cvxpy); intervention magnitude
  environment.py   randomized cluttered environments + NL mission text
  planner.py       SurrogateLLMPlanner, AnthropicLLMPlanner, make_feedback()
  simulator.py     closed loop + deadlock detection + feedback re-planning
  experiments.py   Monte-Carlo study + hallucination sweep
  plots.py         publication-quality figures
  run_all.py       reproduces all figures/tables
  paper_draft.tex  IEEEtran draft (compile on Overleaf; figures auto-included)
  results/         pre-generated figures, CSV, summary
```

## Notes for the paper
- The bibliography in `paper_draft.tex` lists the key related work
  (Ames et al. CBF; Xiao & Belta HOCBF; Misyats et al. multirotor CBF;
  SAGE-LLM; ASMA; SayCan). **Verify the exact venues/arXiv IDs before submission.**
- The robustness study uses a calibrated LLM surrogate; state this explicitly and
  back it with a smaller real-LLM run for the camera-ready, using the backend above.
