"""
run_llm_groq.py
---------------
Run the framework with a REAL LLM via the Groq API, robust to free-tier
daily-token exhaustion.

New in this version
-------------------
* KEY POOL + MID-RUN ROTATION. Provide several keys; when one hits its daily
  quota the runner switches to the next automatically. When the pool is
  empty it PAUSES and asks you to paste a fresh key (or edit the .env file
  and press Enter) — the run then continues from the same environment.
* CHECKPOINT + --resume. After every completed environment the results are
  appended to results_groq/checkpoint.jsonl. If the run dies (quota, Ctrl-C,
  crash), rerun with --resume and finished environments are skipped.
  Environments are regenerated deterministically from --seed0, so a resumed
  run evaluates the exact same envs.
* NO SILENT FALLBACK. The old code substituted a straight-line plan when the
  API failed, silently contaminating results. Now the run checkpoints and
  exits cleanly instead.

Ways to supply keys (all can be combined; duplicates are removed)
-----------------------------------------------------------------
    GROQ_API_KEY=gsk_a              # in .env or the shell
    GROQ_API_KEYS=gsk_a,gsk_b       # comma-separated pool
    GROQ_API_KEY_2=gsk_b            # numbered vars: _2, _3, ...
    --keys gsk_a,gsk_b              # on the command line

How to run
----------
From the directory that CONTAINS the `uav_llm_cbf` folder:

    python3 -m uav_llm_cbf.run_llm_groq --n_envs 30 --obstacles 10
    # ... key dies at env 17, you get a new key tomorrow ...
    python3 -m uav_llm_cbf.run_llm_groq --n_envs 30 --obstacles 10 --resume
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

from . import experiments as ex
from .environment import generate_environment
from .planner import GroqLLMPlanner, LLMUnavailable
from .simulator import simulate, SimConfig
from . import plots

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
def load_env_file(path):
    """Minimal .env loader (so python-dotenv is optional)."""
    if not path or not os.path.exists(path):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path, override=False)
        return
    except Exception:
        pass
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def collect_keys(cli_keys, env_file):
    """Gather all configured keys, in priority order, de-duplicated."""
    keys = []
    if cli_keys:
        keys += [k.strip() for k in cli_keys.split(",") if k.strip()]
    if os.environ.get("GROQ_API_KEY"):
        keys.append(os.environ["GROQ_API_KEY"].strip())
    if os.environ.get("GROQ_API_KEYS"):
        keys += [k.strip() for k in os.environ["GROQ_API_KEYS"].split(",")
                 if k.strip()]
    i = 2
    while os.environ.get(f"GROQ_API_KEY_{i}"):
        keys.append(os.environ[f"GROQ_API_KEY_{i}"].strip())
        i += 1
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class KeyManager:
    """Pool of API keys. rotate() advances to the next key; when the pool is
    exhausted it (a) re-reads the .env file in case you edited it, then
    (b) interactively prompts for a fresh key. Returns False only when the
    user gives up, in which case the runner checkpoints and exits."""

    def __init__(self, keys, env_file=None, interactive=True):
        if not keys:
            raise RuntimeError("No GROQ API key configured.")
        self.keys = list(keys)
        self.index = 0
        self.env_file = env_file
        self.interactive = interactive
        self.exhausted = set()           # keys known to be spent today

    def current(self):
        return self.keys[self.index]

    def _fresh_from_env_file(self):
        """Re-read the .env file; return any key we haven't seen yet."""
        if not self.env_file or not os.path.exists(self.env_file):
            return None
        for line in open(self.env_file):
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            if k.strip().startswith("GROQ_API_KEY"):
                v = v.strip().strip('"').strip("'")
                if v and v not in self.keys:
                    return v
        return None

    def rotate(self):
        self.exhausted.add(self.current())
        # 1) unused key already in the pool?
        for j in range(len(self.keys)):
            if self.keys[j] not in self.exhausted:
                self.index = j
                return True
        # 2) user may have edited the .env file mid-run
        fresh = self._fresh_from_env_file()
        if fresh:
            self.keys.append(fresh)
            self.index = len(self.keys) - 1
            print("  [keys] picked up a new key from the .env file")
            return True
        # 3) interactive pause
        if self.interactive and sys.stdin is not None and sys.stdin.isatty():
            print("\n  [keys] All configured keys are exhausted.")
            print("  Paste a new GROQ API key and press Enter to continue,")
            print(f"  or edit {self.env_file!r} (add GROQ_API_KEY_2=...) and "
                  f"press Enter,")
            print("  or press Enter on an empty line to checkpoint & quit.")
            while True:
                k = input("  new key> ").strip()
                if not k:
                    fresh = self._fresh_from_env_file()
                    if fresh:
                        self.keys.append(fresh)
                        self.index = len(self.keys) - 1
                        print("  [keys] picked up a new key from the .env file")
                        return True
                    return False
                if k in self.exhausted:
                    print("  [keys] that key was already exhausted; try another")
                    continue
                self.keys.append(k)
                self.index = len(self.keys) - 1
                return True
        return False


# ---------------------------------------------------------------------------
class FixedInitialPlanner:
    """Returns a precomputed initial plan; delegates re-planning to the LLM.

    This guarantees all configurations see the *same* initial plan while
    keeping live LLM calls to a minimum (only feedback configs replan)."""
    def __init__(self, initial_plan, llm):
        self._plan = [np.asarray(p, float) for p in initial_plan]
        self._llm = llm
        self.name = llm.name

    def initial_plan(self, env):
        return [p.copy() for p in self._plan]

    def replan(self, env, current_plan, active_idx, feedback):
        return self._llm.replan(env, current_plan, active_idx, feedback)


# ---------------------------------------------------------------------------
def load_checkpoint(path):
    """Return ({env_index: [row, ...]}, n_calls_so_far)."""
    done, calls = {}, 0
    if not os.path.exists(path):
        return done, calls
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        done[rec["env"]] = rec["rows"]
        calls = max(calls, rec.get("n_calls", 0))
    return done, calls


def append_checkpoint(path, env_idx, rows, n_calls):
    with open(path, "a") as f:
        f.write(json.dumps({"env": env_idx, "rows": rows,
                            "n_calls": n_calls}) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=15)
    ap.add_argument("--obstacles", type=int, default=6)
    ap.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    ap.add_argument("--env-file", type=str, default=".env")
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--keys", type=str, default=None,
                    help="comma-separated pool of GROQ API keys")
    ap.add_argument("--resume", action="store_true",
                    help="skip environments already in the checkpoint")
    ap.add_argument("--no-pause", action="store_true",
                    help="never prompt for a key interactively (CI mode)")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    load_env_file(args.env_file)
    keys = collect_keys(args.keys, args.env_file)
    if not keys:
        sys.exit(f"No GROQ API key found. Put GROQ_API_KEY=... in "
                 f"{args.env_file!r}, set GROQ_API_KEYS=k1,k2, or use --keys.")
    km = KeyManager(keys, env_file=args.env_file,
                    interactive=not args.no_pause)

    outdir = args.outdir or os.path.join(HERE, "results_groq")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    ckpt = os.path.join(outdir, "checkpoint.jsonl")

    done = {}
    if args.resume:
        done, prev_calls = load_checkpoint(ckpt)
        print(f"[resume] {len(done)} environment(s) already completed "
              f"({prev_calls} LLM calls in previous session(s)); skipping them.")
    elif os.path.exists(ckpt):
        print(f"[note] {ckpt} exists from a previous run. Use --resume to "
              f"continue it, or delete it to start over.")
        sys.exit(1)

    llm = GroqLLMPlanner(model=args.model, key_manager=km)
    print(f"Using {llm.name} over {args.n_envs} environments "
          f"({args.obstacles} obstacles each), key pool size {len(keys)}.\n")

    short = {"no_filter": "raw", "filter": "filt", "feedback-basic": "fb",
             "feedback-grad": "grad", "feedback-grad-v2": "gradv2"}
    example, example_env = None, None
    aborted = False

    for i in range(args.n_envs):
        if i in done:
            continue
        env = generate_environment(np.random.default_rng(args.seed0 + i),
                                   n_obstacles=args.obstacles)
        try:
            init = llm.initial_plan(env)          # ONE live call per env
            env_rows, env_results = [], {}
            for name, flags in ex.CONFIGS:
                planner = FixedInitialPlanner(init, llm)
                res = simulate(env, planner, SimConfig(**flags))
                env_results[name] = res
                env_rows.append(dict(
                    env=i, config=name,
                    reached=int(res["reached"]), collided=int(res["collided"]),
                    timeout=int(res["timeout"]),
                    path_length=res["path_length"], time=res["time"],
                    replans=res["replans"],
                    mean_intervention=res["mean_intervention"],
                    min_barrier=res["min_barrier"],
                    min_clearance=res["min_clearance"],
                    infeasible_steps=res["infeasible_steps"]))
        except LLMUnavailable as e:
            # an env is only checkpointed when ALL its configs finished, so a
            # partially-run env is simply redone on --resume (envs regenerate
            # deterministically from seed0). Nothing is contaminated.
            print(f"\n[stop] env {i}: LLM unavailable ({e}).")
            print(f"[stop] Progress saved: {len(done)} env(s) in {ckpt}.")
            print(f"[stop] Get a fresh key, then rerun the same command "
                  f"with --resume.")
            aborted = True
            break

        append_checkpoint(ckpt, i, env_rows, llm.n_calls)
        done[i] = env_rows
        if example is None:
            example, example_env = env_results, env
        print(f"  env {i:2d}: "
              + " | ".join(f"{short.get(n, n[:6])}:"
                           f"{'R' if env_results[n]['reached'] else ('C' if env_results[n]['collided'] else 'T')}"
                           for n, _ in ex.CONFIGS)
              + f"   (LLM calls this session: {llm.n_calls}, "
                f"key #{km.index + 1})")

    if not done:
        sys.exit("No environments completed; check your API key/model.")

    # ---- aggregate everything in the checkpoint (this + prior sessions) ----
    rows = [r for i in sorted(done) for r in done[i]]
    ex.save_rows_csv(rows, os.path.join(outdir, "main_study_groq.csv"))
    summary = ex.summarize(rows)
    json.dump(summary, open(os.path.join(outdir, "summary_groq.json"), "w"),
              indent=2)

    plots.metric_bars(summary, figdir)
    if example is not None:
        plots.trajectory_panels(example, example_env, figdir)
        key_cfg = ("feedback-grad-v2" if "feedback-grad-v2" in example
                   else "feedback-grad")
        plots.safety_timeseries(example[key_cfg], dt=0.05, outdir=figdir)

    hdr = (f"{'config':18s} {'success%':>9s} {'collision%':>11s} "
           f"{'timeout%':>9s} {'pathlen':>8s} {'replans':>8s} {'min_clear(m)':>13s}")
    lines = [f"Real LLM: {llm.name}   (envs: {len(done)}/{args.n_envs}, "
             f"LLM calls this session: {llm.n_calls})", hdr, "-" * len(hdr)]
    for name, _ in ex.CONFIGS:
        s = summary[name]
        lines.append(f"{name:18s} {s['success_rate']:9.1f} "
                     f"{s['collision_rate']:11.1f} {s['timeout_rate']:9.1f} "
                     f"{s['mean_path_len']:8.2f} {s['mean_replans']:8.2f} "
                     f"{s['min_clearance']:13.3f}")
    table = "\n".join(lines)
    open(os.path.join(outdir, "summary_table_groq.txt"), "w").write(table + "\n")
    print("\n" + table)
    print(f"\nResults -> {outdir}")
    if aborted:
        sys.exit(2)


if __name__ == "__main__":
    main()
