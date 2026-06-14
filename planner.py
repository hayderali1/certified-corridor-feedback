"""
planner.py
----------
The high-level planner that turns a natural-language mission into a sequence
of UAV waypoints, plus the *control-to-language feedback* translator that is
the novel kernel of this framework.

Two planner backends are provided:

  * SurrogateLLMPlanner  -- a deterministic-but-stochastic stand-in that
        reproduces the *documented spatial-reasoning failure modes* of LLM
        planners (waypoints that clip or sit inside obstacles, over-confident
        straight-line routing). This lets us run large reproducible Monte-Carlo
        studies and ablations without API cost or rate limits, and lets us
        *control* the hallucination rate to characterise robustness.

  * AnthropicLLMPlanner   -- an optional real backend that prompts a Claude
        model through the public API and parses JSON waypoints. It is disabled
        unless an API key is supplied; it is provided so the exact same harness
        can be run with a real LLM for the camera-ready experiments.

Both backends consume the same `feedback` object, which is produced by
`make_feedback` from the CBF safety filter's *intervention magnitude* and a
*deadlock detector*. This is the bidirectional loop: control theory tells the
language model, in words, why its plan was unsafe, and the planner repairs it.
"""

from __future__ import annotations
import os
import json
import numpy as np
from .geometry import bypass_geometry


# ----------------------------------------------------------------------------
# Control -> language feedback
# ----------------------------------------------------------------------------
def make_feedback(blocking_obstacle, current_wp, uav_pos):
    """
    Translate a control-theoretic conflict (the obstacle whose barrier the
    QP fought hardest, returned by the safety filter) into a structured
    natural-language message for the planner.
    """
    c, r = blocking_obstacle
    return {
        "type": "deadlock",
        "text": (
            f"Waypoint near ({current_wp[0]:.1f}, {current_wp[1]:.1f}) is "
            f"blocked: the route forces the UAV into an obstacle of radius "
            f"{r:.1f} centred at ({c[0]:.1f}, {c[1]:.1f}). The safety filter "
            f"is continuously overriding the command and the UAV has stalled "
            f"at ({uav_pos[0]:.1f}, {uav_pos[1]:.1f}). Propose a detour "
            f"waypoint that clears this obstacle."
        ),
        "obstacle": (np.asarray(c, float), float(r)),
        "stalled_at": np.asarray(uav_pos, float),
    }


def make_gradient_feedback(blocking_obstacle, current_wp, uav_pos, goal,
                           obstacles, r_safe, bounds=None):
    """
    NOVEL: control-to-language feedback enriched with the safety filter's local
    geometry. From the barrier gradient and obstacle tangents we derive an open
    escape bearing and control-validated detour waypoints, and verbalize them so
    the LLM re-plans with control-grounded spatial cues rather than guessing.
    """
    c, r = blocking_obstacle
    g = bypass_geometry(uav_pos, goal, blocking_obstacle, obstacles, r_safe,
                        bounds=bounds)
    cand = g["candidates"]
    ctext = "; ".join(f"({w[0]:.1f}, {w[1]:.1f})" for w in cand[:2])
    return {
        "type": "deadlock_gradient",
        "text": (
            f"Waypoint near ({current_wp[0]:.1f}, {current_wp[1]:.1f}) is blocked "
            f"by an obstacle of radius {r:.1f} at ({c[0]:.1f}, {c[1]:.1f}); the "
            f"UAV has stalled at ({uav_pos[0]:.1f}, {uav_pos[1]:.1f}). The safety "
            f"filter's barrier gradient indicates the open direction is toward "
            f"the {g['bearing']}. Control-validated detour waypoints that clear "
            f"all obstacles are: {ctext}. Re-plan to pass through one of these, "
            f"then continue to the goal ({goal[0]:.1f}, {goal[1]:.1f})."
        ),
        "obstacle": (np.asarray(c, float), float(r)),
        "stalled_at": np.asarray(uav_pos, float),
        "candidates": cand,
        "escape_dir": g["escape_dir"],
        "bearing": g["bearing"],
    }


def make_gradient_feedback_v2(blocking_obstacle, current_wp, uav_pos, goal,
                              obstacles, r_safe, bounds=None):
    """
    NOVEL (v2): control-to-language feedback carrying a *line-of-sight
    validated, multi-hop safe corridor* computed from a visibility graph over
    the barrier geometry of ALL obstacles. Unlike v1, every leg of the
    verbalized detour is checked collision-free, so the LLM executes a
    control-certified corridor instead of a point that may be unreachable.
    """
    from .geometry_v2 import bypass_geometry_v2
    c, r = blocking_obstacle
    g = bypass_geometry_v2(uav_pos, goal, blocking_obstacle, obstacles,
                           r_safe, bounds=bounds)
    path = g["path"]
    ctext = " -> ".join(f"({w[0]:.1f}, {w[1]:.1f})" for w in path)
    return {
        "type": "deadlock_gradient_v2",
        "text": (
            f"Waypoint near ({current_wp[0]:.1f}, {current_wp[1]:.1f}) is blocked "
            f"by an obstacle of radius {r:.1f} at ({c[0]:.1f}, {c[1]:.1f}); the "
            f"UAV has stalled at ({uav_pos[0]:.1f}, {uav_pos[1]:.1f}). The safety "
            f"filter's barrier geometry indicates the open direction is toward "
            f"the {g['bearing']}. A safe corridor whose every straight leg has "
            f"been verified collision-free is: {ctext}. Re-plan to follow these "
            f"waypoints in order, then continue to the goal "
            f"({goal[0]:.1f}, {goal[1]:.1f})."
        ),
        "obstacle": (np.asarray(c, float), float(r)),
        "stalled_at": np.asarray(uav_pos, float),
        "candidates": path,
        "path": path,
        "escape_dir": g["escape_dir"],
        "bearing": g["bearing"],
    }


# ----------------------------------------------------------------------------
# Surrogate LLM planner (default backend)
# ----------------------------------------------------------------------------
class SurrogateLLMPlanner:
    """
    Plans a coarse waypoint sequence start -> ... -> goal. With probability
    `halluc_rate` the planner mis-reasons about geometry and emits a waypoint
    that is *unsafe* (sitting on/inside an obstacle) -- exactly the failure the
    safety filter must catch. On feedback it repairs the flagged waypoint by
    routing around the obstacle (tangential detour).
    """

    def __init__(self, rng: np.random.Generator, halluc_rate: float = 0.5,
                 n_waypoints: int = 3):
        self.rng = rng
        self.halluc_rate = float(halluc_rate)
        self.n_waypoints = int(n_waypoints)
        self.name = "surrogate"

    def initial_plan(self, env):
        start, goal = env.start, env.goal
        wps = []
        for k in range(1, self.n_waypoints + 1):
            t = k / (self.n_waypoints + 1)
            wp = (1 - t) * start + t * goal
            # realistic LLM failure: occasionally place the waypoint inside the
            # nearest obstacle (poor metric spatial reasoning), or ignore it.
            if self.rng.random() < self.halluc_rate and env.obstacles:
                # pick obstacle closest to this interpolation point
                c, r = min(env.obstacles,
                           key=lambda o: np.linalg.norm(o[0] - wp))
                jitter = self.rng.uniform(-0.3, 0.3, size=2)
                wp = np.asarray(c, float) + jitter  # waypoint *inside* obstacle
            wps.append(np.asarray(wp, float))
        wps.append(np.asarray(goal, float))
        return wps

    def replan(self, env, current_plan, active_idx, feedback):
        """Repair the plan given control-to-language feedback. Returns a
        from-here plan [detour, goal] (the simulator restarts at index 0).

        Models the contribution faithfully: with GRADIENT feedback the controller
        supplies a validated detour, so the surrogate uses it reliably. With BASIC
        feedback the *LLM* must reason out the detour, so the surrogate inherits
        its spatial-reasoning error (hallucinated detour with prob halluc_rate)."""
        goal = np.asarray(env.goal, float)
        c, r = feedback["obstacle"]
        c = np.asarray(c, float)

        # gradient v2 feedback: controller-validated CORRIDOR -> follow it
        if feedback.get("path"):
            return ([np.asarray(w, float) for w in feedback["path"]]
                    + [goal])

        # gradient v1 feedback: controller-validated candidate -> reliable
        if feedback.get("candidates"):
            return [np.asarray(feedback["candidates"][0], float), goal]

        # basic feedback: LLM does the geometry and errs at halluc_rate
        if self.rng.random() < self.halluc_rate:
            detour = c + self.rng.uniform(-0.4, 0.4, size=2)   # poor detour
            return [detour, goal]
        stalled = feedback["stalled_at"]
        # choose a detour point offset perpendicular to the stalled->goal line,
        # pushed safely outside the blocking obstacle (tangential bypass).
        to_goal = goal - stalled
        nrm = np.linalg.norm(to_goal)
        if nrm < 1e-6:
            to_goal = goal - env.start
            nrm = np.linalg.norm(to_goal) + 1e-6
        dirn = to_goal / nrm
        perp = np.array([-dirn[1], dirn[0]])
        # try both sides, keep the one with more clearance from all obstacles
        clearance_margin = r + 0.9
        candidates = [c + perp * clearance_margin, c - perp * clearance_margin]
        def clearance(pt):
            return min(np.linalg.norm(pt - o[0]) - o[1] for o in env.obstacles)
        detour = max(candidates, key=clearance)
        return [np.asarray(detour, float), np.asarray(goal, float)]


# ----------------------------------------------------------------------------
# Optional real Claude backend (used only if an API key is present)
# ----------------------------------------------------------------------------
class AnthropicLLMPlanner:
    """
    Real LLM backend. Requires ANTHROPIC_API_KEY in the environment and the
    `anthropic` package. Same interface as the surrogate so experiments.py can
    swap it in for camera-ready runs. NOTE: never hard-code a key here.
    """

    def __init__(self, model: str = "claude-opus-4-8", n_waypoints: int = 3):
        self.model = model
        self.n_waypoints = n_waypoints
        self.name = f"anthropic:{model}"
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic  # type: ignore
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _ask(self, prompt):
        client = self._client_lazy()
        msg = client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        data = json.loads(text)
        return [np.asarray(w, float) for w in data["waypoints"]]

    def initial_plan(self, env):
        obs = "; ".join(f"disk r={r:.2f} at ({c[0]:.2f},{c[1]:.2f})"
                        for c, r in env.obstacles)
        prompt = (
            f"You are a UAV mission planner. {env.mission_text}\n"
            f"Obstacles: {obs}.\n"
            f"Return ONLY JSON: {{\"waypoints\": [[x,y], ...]}} with about "
            f"{self.n_waypoints} intermediate waypoints then the goal."
        )
        return self._ask(prompt)

    def replan(self, env, current_plan, active_idx, feedback):
        obs = "; ".join(f"disk r={r:.2f} at ({c[0]:.2f},{c[1]:.2f})"
                        for c, r in env.obstacles)
        prompt = (
            f"You are a UAV mission planner. {env.mission_text}\n"
            f"Obstacles: {obs}.\nPrevious plan failed. Controller feedback: "
            f"{feedback['text']}\nReturn ONLY JSON {{\"waypoints\": [[x,y],...]}} "
            f"that detours around the blocking obstacle to the goal."
        )
        return self._ask(prompt)


# ----------------------------------------------------------------------------
# Groq backend (OpenAI-compatible; fast open models)
# ----------------------------------------------------------------------------
def _wp_feasible(wp, obstacles, buf=0.2):
    """True if waypoint wp is clear of every obstacle (with a small buffer)."""
    wp = np.asarray(wp, float)
    return all(np.linalg.norm(wp - np.asarray(c, float)) > r + buf
               for (c, r) in obstacles)


def _parse_waypoints(text, env):
    """Robustly extract a waypoint list from an LLM text reply."""
    text = text.strip()
    # isolate the JSON object even if wrapped in prose / code fences
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e + 1]
    data = json.loads(text)
    raw = data.get("waypoints") or data.get("plan") or []
    xmin, xmax, ymin, ymax = env.bounds
    wps = []
    for w in raw:
        p = np.asarray(w, float).reshape(-1)[:2]
        p[0] = float(np.clip(p[0], xmin, xmax))
        p[1] = float(np.clip(p[1], ymin, ymax))
        wps.append(p)
    # always finish at the goal so the mission is well-defined
    if not wps or np.linalg.norm(wps[-1] - env.goal) > 1e-3:
        wps.append(np.asarray(env.goal, float))
    return wps


class LLMUnavailable(RuntimeError):
    """Raised when no API key can serve the request. The run driver catches
    this, checkpoints, and exits cleanly so --resume can continue later.
    NOTE: we deliberately removed the old silent straight-line fallback —
    it let exhausted-quota runs masquerade as real LLM results."""
    pass


class GroqLLMPlanner:
    """
    Real LLM backend using Groq's API (OpenAI-compatible). Same interface as
    SurrogateLLMPlanner so the simulator/experiments run unchanged.

    Requires:  pip install groq      and   GROQ_API_KEY in the environment.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 n_waypoints: int = 3, temperature: float = 0.3,
                 max_retries: int = 4, key_manager=None):
        self.model = model
        self.n_waypoints = n_waypoints
        self.temperature = temperature
        self.max_retries = max_retries
        self.name = f"groq:{model}"
        self._client = None
        self.n_calls = 0
        # key_manager: object with .current() -> str and .rotate() -> bool.
        # If None, falls back to the single GROQ_API_KEY env var (old behavior).
        self.key_manager = key_manager

    def _client_lazy(self, force_new=False):
        if self._client is None or force_new:
            from groq import Groq  # type: ignore
            key = (self.key_manager.current() if self.key_manager
                   else os.environ.get("GROQ_API_KEY"))
            if not key:
                raise RuntimeError("GROQ_API_KEY not set in environment.")
            self._client = Groq(api_key=key)
        return self._client

    @staticmethod
    def _is_quota_error(ex):
        """Daily/total token exhaustion or a dead key -> rotating helps.
        Per-minute 429s are NOT quota errors -> sleeping helps."""
        msg = str(ex).lower()
        if any(s in msg for s in ("per day", "tokens per day", "tpd",
                                  "daily", "quota", "billing",
                                  "invalid api key", "invalid_api_key",
                                  "401", "organization_restricted")):
            return True
        code = getattr(ex, "status_code", None)
        return code in (401, 402, 403)

    def _ask(self, prompt, env):
        import time
        last_err = None
        rotations = 0
        attempt = 0
        while True:
            client = self._client_lazy()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": "You are a precise UAV mission planner. "
                                    "Output strictly valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
                self.n_calls += 1
                text = resp.choices[0].message.content
                return _parse_waypoints(text, env)
            except json.JSONDecodeError as ex:
                # parse error: retry the SAME key (it's an LLM hiccup)
                last_err = ex
                attempt += 1
                if attempt >= self.max_retries:
                    raise LLMUnavailable(f"unparseable replies: {ex}")
                time.sleep(1.0)
            except Exception as ex:
                last_err = ex
                if self._is_quota_error(ex):
                    # this key is spent/dead: rotate (or pause for a new one)
                    if self.key_manager and self.key_manager.rotate():
                        rotations += 1
                        self._client_lazy(force_new=True)
                        attempt = 0
                        print(f"  [groq] key exhausted -> switched to key "
                              f"#{self.key_manager.index + 1}; resuming")
                        continue
                    raise LLMUnavailable(f"quota exhausted, no key left: {ex}")
                # transient / per-minute rate limit: back off, same key
                attempt += 1
                if attempt >= self.max_retries:
                    # persistent failure: treat like quota (maybe rotate)
                    if self.key_manager and self.key_manager.rotate():
                        rotations += 1
                        self._client_lazy(force_new=True)
                        attempt = 0
                        continue
                    raise LLMUnavailable(f"giving up after retries: {ex}")
                time.sleep(2.0 * attempt)

    def initial_plan(self, env):
        obs = "; ".join(f"a disk of radius {r:.2f} centred at "
                        f"({c[0]:.2f}, {c[1]:.2f})" for c, r in env.obstacles)
        prompt = (
            f"{env.mission_text}\n"
            f"The arena spans x in [{env.bounds[0]:.0f},{env.bounds[1]:.0f}], "
            f"y in [{env.bounds[2]:.0f},{env.bounds[3]:.0f}].\n"
            f"Obstacles (must be avoided): {obs}.\n"
            f"Plan about {self.n_waypoints} intermediate waypoints, then the "
            f"goal. Respond with ONLY this JSON: "
            f"{{\"waypoints\": [[x,y], [x,y], ...]}}."
        )
        return self._ask(prompt, env)

    def replan(self, env, current_plan, active_idx, feedback):
        obs = "; ".join(f"a disk of radius {r:.2f} centred at "
                        f"({c[0]:.2f}, {c[1]:.2f})" for c, r in env.obstacles)
        prompt = (
            f"{env.mission_text}\n"
            f"The arena spans x in [{env.bounds[0]:.0f},{env.bounds[1]:.0f}], "
            f"y in [{env.bounds[2]:.0f},{env.bounds[3]:.0f}].\n"
            f"Obstacles: {obs}.\n"
            f"Your previous plan failed. Controller feedback: {feedback['text']}\n"
            f"Provide a corrected detour. Respond with ONLY this JSON: "
            f"{{\"waypoints\": [[x,y], ...]}} ending at the goal."
        )
        wps = self._ask(prompt, env)
        # gradient feedback: certify the LLM's route. v1 only checked the
        # FIRST waypoint by point clearance; now every waypoint AND every
        # straight leg from the stall pose is verified. If the route is
        # unsafe, prepend the control-validated corridor so the immediate
        # detour is guaranteed feasible.
        cands = feedback.get("path") or feedback.get("candidates")
        if cands:
            def _route_ok(route):
                pts = [np.asarray(feedback["stalled_at"], float)] + list(route)
                for w in route:
                    if not _wp_feasible(w, env.obstacles):
                        return False
                for a, b in zip(pts[:-1], pts[1:]):
                    d = b - a
                    L2 = float(d @ d)
                    for (c2, r2) in env.obstacles:
                        c2 = np.asarray(c2, float)
                        t = 0.0 if L2 == 0 else float(np.clip((c2 - a) @ d / L2, 0, 1))
                        if np.linalg.norm(a + t * d - c2) < r2 + 0.2:
                            return False
                return True
            if not _route_ok(wps):
                wps = [np.asarray(w, float) for w in cands] + list(wps)
        return wps
