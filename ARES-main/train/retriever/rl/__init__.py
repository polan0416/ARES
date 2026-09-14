from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parent

__all__ = ["RL_ROOT", "PROJECT_ROOT"]
