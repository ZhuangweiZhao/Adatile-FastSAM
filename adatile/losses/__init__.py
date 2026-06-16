"""AS-FastSAM loss modules.

Clean decomposition of UnifiedLoss:
  seg_loss.py    — BinarySegLoss, MultiClassSegLoss
  spm_loss.py    — DensityLoss, TopKLoss     (importance ranking supervision)
  budget_loss.py — FixedBudget, LearnableBudget (activation rate control)
"""

from adatile.losses.seg_loss import BinarySegLoss, MultiClassSegLoss, SegLoss
from adatile.losses.spm_loss import DensityLoss, TopKLoss
from adatile.losses.budget_loss import FixedBudget, LearnableBudget
from adatile.losses.unified import UnifiedLoss
