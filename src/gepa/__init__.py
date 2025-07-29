"""
GEPA: Reflective Prompt Evolution

An inference-agnostic prompt optimization toolkit that uses natural language reflection
to learn high-level rules from trial and error, outperforming traditional RL methods
with significantly fewer rollouts.

Based on the research paper:
"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
by Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, 
Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, 
Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, 
Matei Zaharia, and Omar Khattab (2025).

arXiv preprint arXiv:2507.19457
"""

__version__ = "0.1.0"
__author__ = "Nick Paterno"
__paper__ = "https://arxiv.org/abs/2507.19457"

from .core.optimizer import GEPAOptimizer
from .core.system import CompoundAISystem
from .inference.factory import InferenceFactory
from .evaluation.metrics import Metric, ExactMatch, F1Score
from .config import GEPAConfig

__all__ = [
    "GEPAOptimizer",
    "CompoundAISystem", 
    "InferenceFactory",
    "Metric",
    "ExactMatch",
    "F1Score",
    "GEPAConfig",
]