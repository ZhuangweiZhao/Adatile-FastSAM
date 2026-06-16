"""Lightweight run logger — all values saved to disk.

Creates: outputs/<name>_<timestamp>/
  ├── config.json      — all CLI args
  ├── system_info.txt  — GPU, CUDA, PyTorch, git
  ├── train.log        — full console output (dual: file + stdout)
  ├── metrics.csv      — step,loss,iou,dice,... (dynamic columns)
  ├── memory.log       — step,alloc_mb,reserved_mb,peak_mb
  ├── results.json     — all aggregated results + best metrics
  └── summary.txt      — best metrics + duration (human-readable)
"""

from __future__ import annotations

import csv, json, logging, os, sys, time, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class _TeeWriter:
    """Writes to both a file and the original stdout."""
    def __init__(self, filepath: str, original_stdout):
        self.file = open(filepath, "w", encoding="utf-8", buffering=1)
        self.original = original_stdout
    def write(self, msg):
        self.file.write(msg)
        self.original.write(msg)
    def flush(self):
        self.file.flush()
        self.original.flush()
    def close(self):
        self.file.close()


class RunLogger:
    """Experiment logger — saves ALL values to disk.

    Usage:
        log = RunLogger("outputs", "ablation", vars(args))
        log.start()
        log.log(step=100, loss=0.5, iou=0.7)
        log.log_table("ablation", list_of_dicts)  # structured tables
        log.finish()
    """

    def __init__(self, root: str, name: str, args_dict: Optional[Dict] = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{name}_{ts}"
        self.args = args_dict or {}
        self._csv_file = None
        self._csv_writer = None
        self._start_time = None
        self._best: Dict[str, Any] = {}
        self._fieldnames: List[str] = []
        self._tee: Optional[_TeeWriter] = None
        self._tables: Dict[str, List[Dict]] = {}

    def start(self, extra_info: Optional[Dict] = None):
        """Create directory, initialize all log files."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

        # Dual output: file + console
        self._tee = _TeeWriter(str(self.run_dir / "train.log"), sys.stdout)
        sys.stdout = self._tee

        # config.json
        info = {"started": datetime.now().isoformat(), **self.args}
        if extra_info: info.update(extra_info)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

        self._save_system_info()
        print(f"[RunLogger] {self.run_dir}")

    def elapsed(self) -> float:
        """Seconds since start()."""
        return time.time() - (self._start_time or time.time())

    def log(self, step: int, **metrics):
        """Write one row to metrics.csv. Auto-adds elapsed time."""
        if self._csv_writer is None:
            self._fieldnames = ["step"] + sorted(metrics.keys())
            self._csv_file = open(self.run_dir / "metrics.csv", "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

        # Expand fieldnames if new keys appear
        new_keys = [k for k in metrics if k not in self._fieldnames]
        if new_keys:
            self._csv_file.close()
            self._fieldnames += sorted(new_keys)
            old_path = self.run_dir / "metrics.csv"
            old_rows = []
            if old_path.exists():
                with open(old_path, "r") as f:
                    for r in csv.DictReader(f): old_rows.append(r)
            self._csv_file = open(old_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
            for r in old_rows: self._csv_writer.writerow(r)

        row: Dict[str, Any] = {"step": step}
        for k in self._fieldnames:
            if k == "step": continue
            row[k] = _fmt(metrics.get(k, ""))
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_table(self, name: str, rows: List[Dict], save_csv: bool = True):
        """Save a structured table (e.g. ablation results) to JSON + optional CSV.

        Args:
            name: table name (e.g. "ablation", "fewshot_comparison")
            rows: list of dicts, each dict is one row
        """
        self._tables[name] = rows
        # Save as JSON immediately
        with open(self.run_dir / f"{name}.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)
        # Also save as CSV
        if save_csv and rows:
            with open(self.run_dir / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    def log_memory(self, step: int):
        """Record GPU memory snapshot."""
        if not torch.cuda.is_available(): return
        mem_path = self.run_dir / "memory.log"
        exists = mem_path.exists()
        with open(mem_path, "a") as f:
            if not exists: f.write("step,alloc_mb,reserved_mb,peak_mb\n")
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            f.write(f"{step},{alloc:.1f},{reserved:.1f},{peak:.1f}\n")

    def log_best(self, **metrics):
        """Update best metrics (saved to summary and results.json on finish)."""
        self._best.update(metrics)

    def finish(self):
        """Write summary, results.json, and close all files."""
        elapsed = time.time() - (self._start_time or 0)
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        # summary.txt
        lines = [f"Experiment: {self.run_dir.name}",
                 f"Completed:  {datetime.now().isoformat()}",
                 f"Duration:   {h}h{m:02d}m{s:02d}s"]
        for k, v in self._best.items():
            lines.append(f"Best {k}: {_fmt(v)}")
        lines.append(f"Peak GPU:  {torch.cuda.max_memory_allocated()/1024**2:.0f} MB"
                     if torch.cuda.is_available() else "GPU: N/A")
        with open(self.run_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        # results.json — all best + tables
        results = {"config": self.args, "duration_s": elapsed,
                   "best": {k: _fmt(v) for k, v in self._best.items()},
                   "tables": self._tables}
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self._csv_file: self._csv_file.close()
        if self._tee:
            sys.stdout = self._tee.original
            self._tee.close()

        print(f"[RunLogger] Done. Best: {self._best} → {self.run_dir}")

    def exception(self, exc: Exception):
        """Log exception traceback."""
        import traceback
        with open(self.run_dir / "exception.log", "w") as f:
            f.write(traceback.format_exc())

    def _save_system_info(self):
        lines = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                lines.append(f"GPU[{i}]: {name} ({total:.1f} GB)")
        else:
            lines.append("GPU: None")
        lines.append(f"CUDA: {torch.version.cuda or 'N/A'}")
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"Python: {sys.version.split()[0]}")
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()[:8]
            lines.append(f"Git: {commit}")
        except Exception: pass
        with open(self.run_dir / "system_info.txt", "w") as f:
            f.write("\n".join(lines) + "\n")


def _fmt(v: Any) -> str:
    if isinstance(v, float): return f"{v:.6f}"
    return str(v)
