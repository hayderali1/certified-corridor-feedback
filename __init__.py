"""LLM-guided UAV navigation with HOCBF safety filtering and
control-to-language feedback re-planning."""
__version__ = "0.4-bugfix-active-reset+snap"

from .dynamics import DoubleIntegratorUAV
from .controller import PDTrackingController
from .cbf_filter import HOCBFSafetyFilter
from .environment import generate_environment, Environment
from .planner import (SurrogateLLMPlanner, AnthropicLLMPlanner,
                      GroqLLMPlanner, make_feedback, make_gradient_feedback)
from .simulator import simulate, SimConfig

__all__ = [
    "DoubleIntegratorUAV", "PDTrackingController", "HOCBFSafetyFilter",
    "generate_environment", "Environment", "SurrogateLLMPlanner",
    "AnthropicLLMPlanner", "GroqLLMPlanner", "make_feedback",
    "simulate", "SimConfig",
]
