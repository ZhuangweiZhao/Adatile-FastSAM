"""Training hook system.

Hooks are callbacks that execute at specific points during training:
    - before_train / after_train
    - before_epoch / after_epoch
    - before_step / after_step
    - before_eval / after_eval

Modeled after Detectron2's HookBase with extensions for mixed precision
and few-shot evaluation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class HookBase(ABC):
    """Base class for training hooks."""

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_epoch(self) -> None:
        pass

    def after_epoch(self) -> None:
        pass

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass

    def before_eval(self) -> None:
        pass

    def after_eval(self) -> None:
        pass


class LRSchedulerHook(HookBase):
    """Updates learning rate scheduler at each step."""

    def after_step(self) -> None:
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()


class LoggingHook(HookBase):
    """Logs training metrics at intervals."""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval

    def after_step(self) -> None:
        trainer = self.trainer
        if trainer.global_step % self.log_interval == 0:
            metrics = {m.name: m.avg for m in trainer.meters}
            metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
            metrics["step"] = trainer.global_step
            trainer.logger.info(
                f"Step [{trainer.global_step}/{trainer.max_steps}] "
                + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )


class CheckpointHook(HookBase):
    """Saves checkpoints at regular intervals."""

    def __init__(self, save_interval: int = 5000):
        self.save_interval = save_interval

    def after_step(self) -> None:
        trainer = self.trainer
        if trainer.global_step % self.save_interval == 0:
            trainer.checkpoint_manager.save(
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                step=trainer.global_step,
                epoch=trainer.current_epoch,
            )


class EvalHook(HookBase):
    """Runs evaluation at regular intervals."""

    def __init__(self, eval_interval: int = 1000, key_metric: str = "coco_mask_ap"):
        self.eval_interval = eval_interval
        self.key_metric = key_metric

    def after_step(self) -> None:
        trainer = self.trainer
        if (
            trainer.val_loader is not None
            and trainer.global_step % self.eval_interval == 0
        ):
            trainer.logger.info(f"Running evaluation at step {trainer.global_step}...")
            metrics = trainer.evaluate()
            is_best = trainer.checkpoint_manager.save_best(
                trainer.model,
                metrics,
                key_metric=self.key_metric,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                step=trainer.global_step,
                epoch=trainer.current_epoch,
            )
            if is_best:
                trainer.logger.info(f"New best {self.key_metric}: {metrics[self.key_metric]:.4f}")
            trainer.model.train()


class TensorBoardHook(HookBase):
    """Writes scalars and histograms to TensorBoard."""

    def __init__(self, log_dir: str, flush_secs: int = 30):
        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.log_dir), flush_secs=self.flush_secs)
        return self._writer

    def after_step(self) -> None:
        trainer = self.trainer
        if trainer.global_step % trainer.log_interval == 0:
            for meter in trainer.meters:
                self.writer.add_scalar(
                    f"train/{meter.name}", meter.avg, trainer.global_step
                )

    def after_train(self) -> None:
        if self._writer is not None:
            self._writer.close()


class DiagnosticsHook(HookBase):
    """Unified diagnostics hook: memory + tiles + tokens + router + latency.

    Collects per-step stats from all pipeline stages, writes to CSV
    and TensorBoard. Designed to pinpoint exactly WHERE OOM, token
    explosion, or router collapse occurs.
    """

    def __init__(self, output_dir: str = "outputs",
                 log_interval: int = 50,
                 profile_steps: Optional[list] = None):
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.profile_steps = profile_steps or []
        self._profiler = None

        # Lazy init after trainer is attached
        self._dc = None
        self._mem = None
        self._oom = None

    @property
    def dc(self):
        if self._dc is None:
            from adatile.utils import DiagnosticsCollector
            self._dc = DiagnosticsCollector(str(self.output_dir))
        return self._dc

    @property
    def mem(self):
        if self._mem is None:
            from adatile.utils import MemoryLogger
            self._mem = MemoryLogger(
                str(self.output_dir / "logs" / "memory.log"),
                str(self.output_dir / "logs" / "memory.csv"),
            )
        return self._mem

    @property
    def oom(self):
        if self._oom is None:
            from adatile.utils import OOMGuard
            self._oom = OOMGuard()
        return self._oom

    def before_step(self) -> None:
        trainer = self.trainer
        step = trainer.global_step
        self.mem.set_step(step)
        self.mem.reset_peak()

        # Torch profiler for specific steps
        if step in self.profile_steps:
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._profiler.__enter__()
            trainer.logger.info("[Profiler] Started profiling at step %d", step)

    def after_step(self) -> None:
        trainer = self.trainer
        step = trainer.global_step

        # Stop profiler if active
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            trace_path = str(self.output_dir / "profiles" / f"trace_step{step}.json")
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(trace_path)
            trainer.logger.info("[Profiler] Trace saved to %s", trace_path)
            self._profiler = None

        # Flush memory CSV and summary periodically
        if step % self.log_interval == 0 and step > 0:
            self.mem.save_csv()
            self.dc.step_forward()
            self.oom.update_state(
                step=step, epoch=trainer.current_epoch,
                tiles=getattr(self.dc.tile, 'num_tiles', 0) if self.dc.tile else 0,
                tokens=getattr(self.dc.token, 'tokens_before', 0) if self.dc.token else 0,
            )

            # Log summary
            summary = self.dc.get_summary()
            if summary:
                trainer.logger.info("\n" + summary)

    def after_train(self) -> None:
        self.dc.close()
        self.mem.save_csv()

    def after_epoch(self) -> None:
        trainer = self.trainer
        peak_mb = self.mem.get_peak()
        trainer.logger.info(
            "[Memory] Epoch %d peak: %.0f MB",
            trainer.current_epoch, peak_mb,
        )

