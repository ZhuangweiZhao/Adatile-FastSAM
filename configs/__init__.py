"""Experiment configuration presets.

Each config file defines a Config() with dataset- and task-specific overrides.
Usage:
    cfg = Config.from_yaml("configs/isaid.py")
    cfg = get_isaid_config()
"""

from adatile.config import Config

from .default import get_default_config
from .isaid import get_isaid_config

__all__ = ["Config", "get_default_config", "get_isaid_config"]
