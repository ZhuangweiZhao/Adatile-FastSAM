"""Few-shot experiment configurations."""

from .one_shot import get_1shot_config
from .five_shot import get_5shot_config
from .ten_shot import get_10shot_config

__all__ = ["get_1shot_config", "get_5shot_config", "get_10shot_config"]
