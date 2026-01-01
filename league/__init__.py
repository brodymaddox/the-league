"""The League - Multi-agent RL Competition Framework"""

from .config import load_config, get_team
from .train import train_team
from .compete import run_match

__all__ = ["load_config", "get_team", "train_team", "run_match"]
