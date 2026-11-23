"""MSCA ontology-constrained multi-agent system."""

from .config import ProjectConfig
from .runtime import build_workflow

__all__ = ["ProjectConfig", "build_workflow"]
__version__ = "0.2.0"
