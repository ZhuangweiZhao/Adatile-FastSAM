"""Experiment logging system for AdaTile-FastSAM.

Auto-creates timestamped experiment directories, logs metrics to CSV,
tracks GPU memory, records system info, handles exceptions gracefully.

Paper-grade logging (NEW):
    PaperLogger — enhanced metrics, anomaly detection, auto-visualization
    AnomalyDetector — NaN/gradient/collapse detection
    LogVisualizer — auto-generate publication-ready plots

Usage:
    from adatile.logging import RunLogger          # default, all scripts
    from adatile.logging import PaperLogger         # paper-grade tracking
    from adatile.logging import ExperimentLogger    # legacy Trainer
"""

from adatile.logging.experiment_logger import ExperimentLogger
from adatile.logging.experiment_hook import ExperimentHook
from adatile.logging.run_logger import RunLogger
from adatile.logging.paper_logger import PaperLogger
from adatile.logging.anomaly_detector import AnomalyDetector
from adatile.logging.visualizer import LogVisualizer

__all__ = [
    "RunLogger",
    "PaperLogger",
    "ExperimentLogger",
    "ExperimentHook",
    "AnomalyDetector",
    "LogVisualizer",
]
