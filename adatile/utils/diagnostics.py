"""Unified diagnostics: tile stats, token stats, router stats, decoder stats.

Collects per-step statistics from all pipeline stages and outputs
to structured logs, CSV files, and TensorBoard.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("adatile.diagnostics")


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class TileStats:
    """Per-step tile planner statistics."""
    step: int = 0
    num_tiles: int = 0
    avg_tile_size: float = 0.0
    max_tile_size: int = 0
    size_distribution: Dict[int, int] = field(default_factory=dict)
    entropy: float = 0.0
    skip_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step, "num_tiles": self.num_tiles,
            "avg_tile_size": round(self.avg_tile_size, 1),
            "max_tile_size": self.max_tile_size,
            "entropy": round(self.entropy, 4),
            "skip_ratio": round(self.skip_ratio, 4),
            "size_dist": self.size_distribution,
        }


@dataclass
class TokenStats:
    """Per-step token statistics."""
    step: int = 0
    tokens_before: int = 0   # before router
    tokens_after: int = 0    # after routing (active)
    compression_ratio: float = 1.0
    mean_tokens_per_tile: float = 0.0
    max_tokens_per_tile: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step, "before": self.tokens_before,
            "after": self.tokens_after,
            "compression": round(self.compression_ratio, 2),
            "mean_per_tile": round(self.mean_tokens_per_tile, 1),
            "max_per_tile": self.max_tokens_per_tile,
        }


@dataclass
class RouterStats:
    """Per-step router statistics."""
    step: int = 0
    skip_ratio: float = 0.0
    linear_ratio: float = 0.0
    lowrank_ratio: float = 0.0
    full_ratio: float = 0.0
    mean_weight: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "skip": round(self.skip_ratio, 4),
            "linear": round(self.linear_ratio, 4),
            "lowrank": round(self.lowrank_ratio, 4),
            "full": round(self.full_ratio, 4),
            "mean_weight": round(self.mean_weight, 4),
            "entropy": round(self.entropy, 4),
        }


@dataclass
class DecoderStats:
    """Per-step decoder statistics."""
    step: int = 0
    num_instances: int = 0
    proto_shape: str = ""
    mask_shape: str = ""
    mean_score: float = 0.0
    oom_risk: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step, "num_instances": self.num_instances,
            "proto": self.proto_shape, "mask_shape": self.mask_shape,
            "mean_score": round(self.mean_score, 4),
            "oom_risk": self.oom_risk,
        }


@dataclass
class LatencyStats:
    """Per-step latency breakdown (ms)."""
    backbone: float = 0
    adaspm: float = 0
    tokenizer: float = 0
    router: float = 0
    decoder: float = 0
    loss: float = 0
    total: float = 0

    def to_dict(self) -> dict:
        return {
            "backbone_ms": round(self.backbone, 2),
            "adaspm_ms": round(self.adaspm, 2),
            "tokenizer_ms": round(self.tokenizer, 2),
            "router_ms": round(self.router, 2),
            "decoder_ms": round(self.decoder, 2),
            "loss_ms": round(self.loss, 2),
            "total_ms": round(self.total, 2),
        }


# ── Unified Diagnostics Collector ────────────────────────────────────


class DiagnosticsCollector:
    """Collects and logs per-step pipeline statistics.

    Usage:
        dc = DiagnosticsCollector("outputs")
        dc.log_tiles(128, {384: 45, 768: 50, ...})
        dc.log_tokens(16384, 2048)
        dc.log_router({"skip": 0.42, "linear": 0.35, ...})
        dc.log_decoder(84, "[32,192,192]", "[84,192,192]")
        dc.step()
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.tb_dir = self.output_dir / "tensorboard"
        self.debug_dir = self.output_dir / "debug"
        for d in [self.log_dir, self.tb_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._step = 0
        self._start_time = time.time()

        # Current-step records
        self.tile: Optional[TileStats] = None
        self.token: Optional[TokenStats] = None
        self.router: Optional[RouterStats] = None
        self.decoder: Optional[DecoderStats] = None
        self.latency = LatencyStats()

        # CSV writers (lazy)
        self._csv_writers: Dict[str, Any] = {}

        # TensorBoard writer
        self._tb_writer = None

    @property
    def step(self) -> int:
        return self._step

    @property
    def tb_writer(self):
        if self._tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(str(self.tb_dir))
        return self._tb_writer

    # ── Logging methods ──────────────────────────────────────────

    def log_tiles(self, num_tiles: int, size_dist: Dict[int, int],
                  skip_ratio: float = 0.0):
        """Record tile planner output."""
        total = max(sum(size_dist.values()), 1)
        avg_size = sum(k * v for k, v in size_dist.items()) / total
        max_size = max(size_dist.keys()) if size_dist else 0

        # Shannon entropy
        entropy = 0.0
        for count in size_dist.values():
            if count > 0:
                p = count / total
                entropy -= p * (p ** 0.5)  # simplified

        self.tile = TileStats(
            step=self._step, num_tiles=num_tiles,
            avg_tile_size=avg_size, max_tile_size=max_size,
            size_distribution=size_dist,
            entropy=entropy, skip_ratio=skip_ratio,
        )

        # Log
        dist_str = " ".join(f"{sz}={cnt}" for sz, cnt in sorted(size_dist.items()))
        logger.info(
            "[TilePlanner] tiles=%d skip=%.1f%% sizes=[%s]",
            num_tiles, skip_ratio * 100, dist_str,
        )

    def log_tokens(self, before: int, after: int, num_tiles: int = 1):
        """Record token counts."""
        compression = before / max(after, 1)
        self.token = TokenStats(
            step=self._step, tokens_before=before, tokens_after=after,
            compression_ratio=compression,
            mean_tokens_per_tile=before / max(num_tiles, 1),
            max_tokens_per_tile=before,
        )
        logger.info(
            "[Tokens] before=%d after=%d compression=%.1fx mean/tile=%.1f",
            before, after, compression, self.token.mean_tokens_per_tile,
        )

    def log_router(self, level_dist: Dict[int, float], mean_weight: float = 0,
                   entropy: float = 0):
        """Record router decisions."""
        self.router = RouterStats(
            step=self._step,
            skip_ratio=level_dist.get(0, 0),
            linear_ratio=level_dist.get(1, 0),
            lowrank_ratio=level_dist.get(2, 0),
            full_ratio=level_dist.get(3, 0),
            mean_weight=mean_weight,
            entropy=entropy,
        )
        logger.info(
            "[Router] skip=%.0f%% linear=%.0f%% lowrank=%.0f%% full=%.0f%% weight=%.3f",
            level_dist.get(0, 0) * 100, level_dist.get(1, 0) * 100,
            level_dist.get(2, 0) * 100, level_dist.get(3, 0) * 100,
            mean_weight,
        )

    def log_decoder(self, num_instances: int, proto_shape: str,
                    mask_shape: str, mean_score: float = 0):
        """Record decoder output."""
        # Crop-level masks: each ≤256×256. 2000×256×256×4 = 500 MB.
        # Only flag truly dangerous counts (> 2000 instances).
        oom_risk = num_instances > 2000
        self.decoder = DecoderStats(
            step=self._step, num_instances=num_instances,
            proto_shape=proto_shape, mask_shape=mask_shape,
            mean_score=mean_score, oom_risk=oom_risk,
        )
        logger.info(
            "[Decoder] proto=%s instances=%d mask=%s%s",
            proto_shape, num_instances, mask_shape,
            " ⚠OOM_RISK" if oom_risk else "",
        )

    def log_latency(self, stage: str, ms: float):
        """Record per-stage latency."""
        setattr(self.latency, stage, ms)
        self.latency.total += ms

    def log_loss(self, loss_dict: Dict[str, float]):
        """Record loss values."""
        parts = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
        logger.info("[Loss] %s", parts)

    def log_oom_risk(self, name: str, shape: tuple, estimated_mb: float):
        """Log OOM risk warning."""
        logger.warning(
            "[OOM_RISK] %s shape=%s estimated=%.0fMB",
            name, shape, estimated_mb,
        )

    # ── Step / epoch management ──────────────────────────────────

    def step_forward(self):
        """Advance to next step, flush CSV, write TensorBoard."""
        self._step += 1

        # TensorBoard
        if self._step % 10 == 0:
            w = self.tb_writer
            if self.tile:
                w.add_scalar("tiles/num", self.tile.num_tiles, self._step)
                w.add_scalar("tiles/skip_ratio", self.tile.skip_ratio, self._step)
            if self.token:
                w.add_scalar("tokens/before", self.token.tokens_before, self._step)
                w.add_scalar("tokens/after", self.token.tokens_after, self._step)
                w.add_scalar("tokens/compression", self.token.compression_ratio, self._step)
            if self.router:
                w.add_scalar("router/skip", self.router.skip_ratio, self._step)
                w.add_scalar("router/full", self.router.full_ratio, self._step)
            if self.decoder:
                w.add_scalar("decoder/instances", self.decoder.num_instances, self._step)

        # CSV flush every 50 steps
        if self._step % 50 == 0:
            self._flush_csv()

    def _flush_csv(self):
        """Write accumulated stats to CSV files."""
        records = {
            "tile": self.tile,
            "token": self.token,
            "router": self.router,
            "decoder": self.decoder,
        }
        for name, record in records.items():
            if record is None:
                continue
            path = self.log_dir / f"{name}_stats.csv"
            write_header = not path.exists()
            with open(path, "a", newline="") as f:
                d = record.to_dict()
                writer = csv.DictWriter(f, fieldnames=list(d.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(d)

        # Latency CSV
        if self.latency.total > 0:
            path = self.log_dir / "latency.csv"
            write_header = not path.exists()
            d = self.latency.to_dict()
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(d.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(d)

    def get_summary(self) -> str:
        """Return a formatted summary of current step."""
        lines = [
            f"── Step {self._step} ──",
            f"  Tiles:  {self.tile.num_tiles if self.tile else '?'} "
            f"(skip={self.tile.skip_ratio:.1%})" if self.tile else "",
            f"  Tokens: {self.token.tokens_before}→{self.token.tokens_after if self.token else '?'} "
            f"({self.token.compression_ratio:.1f}x)" if self.token else "",
            f"  Router: S={self.router.skip_ratio:.0%} "
            f"L={self.router.linear_ratio:.0%} "
            f"F={self.router.full_ratio:.0%}" if self.router else "",
            f"  Decoder: {self.decoder.num_instances} instances" if self.decoder else "",
        ]
        return "\n".join(line for line in lines if line)

    def close(self):
        """Flush all data and close writers."""
        self._flush_csv()
        if self._tb_writer:
            self._tb_writer.close()
            self._tb_writer = None
