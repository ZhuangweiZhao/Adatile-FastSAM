#!/usr/bin/env python
"""AS-FastSAM Unified Trainer — importable & CLI.

Usage (CLI):
    python tools/train_as_fastsam.py                    # Stage A: Backbone + Decoder
    python tools/train_as_fastsam.py --use-spm          # Stage B: + Ada-SPM
    python tools/train_as_fastsam.py --use-spm --use-planner  # Stage C: + sparse eval

Usage (import):
    from tools.train_as_fastsam import main

All shared training infrastructure has been extracted to adatile.* modules:
    adatile.losses.unified        → UnifiedLoss
    adatile.engine.builder        → build_components, collect_params, save_checkpoint
    adatile.evaluation.sparse_eval → sparse_eval, split_support_query
    adatile.engine.experiment_runner → ExperimentRunner
"""

import argparse, os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.backbone.fastsam_hook import FastSAMHookBackbone
from adatile.decoder.light_decoder import LightDecoder
from adatile.sparse.light_spm import LightSPM
from adatile.datasets.universal import UniversalDataset
from adatile.logging import RunLogger, PaperLogger
from adatile.utils.early_stop import EarlyStopping
from adatile.losses import UnifiedLoss
from adatile.engine import build_components, collect_params, save_checkpoint
from adatile.engine import ExperimentRunner
from adatile.evaluation import sparse_eval


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AS-FastSAM Unified Trainer")
    p.add_argument("--dataset", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--use-spm", action="store_true")
    p.add_argument("--spm-type", type=str, default="light",
                   choices=["light", "lite", "full", "density_only"],
                   help="SPM architecture variant")
    p.add_argument("--spm-mode", type=str, default="topk", choices=["topk","density"],
                   help="topk=per-image ranking, density=absolute density regression")
    p.add_argument("--use-planner", action="store_true")
    p.add_argument("--n-shot", type=int, default=-1)
    p.add_argument("--episodic", action="store_true")
    p.add_argument("--keep-ratio", type=float, default=0.15)
    p.add_argument("--budget-mode", type=str, default="ratio",
                   choices=["ratio", "entropy", "learnable"])
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=None,
                   help="None=auto-detect, 1=binary, >1=multi-class")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    stage = "A"
    if args.use_spm:
        stage = "B"
    if args.use_spm and args.use_planner:
        stage = "C"

    # PaperLogger: enhanced logging with loss decomposition, anomaly detection, auto-visualization
    plog = PaperLogger("outputs", f"stage{stage}", vars(args), log_interval=max(1, args.max_steps // 20))
    log = plog.runner  # backward-compatible RunLogger access

    img_sz = None if args.image_size == 0 else args.image_size
    img_sz_t = None if img_sz is None else (img_sz, img_sz)
    full_ds = UniversalDataset(args.dataset, split="train", image_size=img_sz_t, num_classes=None)
    args.num_classes = full_ds.num_classes  # sync auto-detected
    val_ds = UniversalDataset(args.dataset, split="val", image_size=img_sz_t, num_classes=args.num_classes)
    print(f"Layout: {full_ds.layout}, num_classes={full_ds.num_classes}")

    vl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    batch_sz = 1 if img_sz is None else 4

    _best_overall_dice = 0.0
    _best_overall_step = 0
    _n_params_total = 0

    for seed in range(args.seeds):
        torch.manual_seed(seed)

        # Build components using shared builder
        backbone, decoder, spm = build_components(args, device, args.num_classes)

        # Save param count before training (for PaperLogger finish)
        _n_params_total = sum(p.numel() for p in backbone.parameters())
        _n_params_total += sum(p.numel() for p in decoder.parameters())
        if spm is not None:
            _n_params_total += sum(p.numel() for p in spm.parameters())

        loss_fn = UnifiedLoss(
            use_spm=args.use_spm,
            num_classes=args.num_classes,
            spm_mode=args.spm_mode,
            budget_target=args.keep_ratio,
            budget_mode=args.budget_mode,
            lambda_reg=args.lambda_reg,
        )

        # Start paper-grade logging
        if seed == 0:
            plog.start(backbone=backbone, decoder=decoder, spm=spm, loss_fn=loss_fn)

        # Setup runner
        runner = ExperimentRunner(
            device=device,
            max_steps=args.max_steps,
            lr=args.lr,
            patience=15,
            mode="max",
        )
        runner.setup(backbone, decoder, spm, loss_fn, full_ds, vl)

        # Wire logging callbacks
        def make_log_cb(seed, log):
            def log_cb(step, val_metrics):
                log.log(
                    step,
                    loss=0,  # filled by step callback
                    iou=val_metrics.get("train_iou", 0),
                    dice=val_metrics.get("train_dice", 0),
                    imp_mean=val_metrics.get("train_imp_mean", 0),
                    coverage=val_metrics.get("train_coverage", 0),
                    val_iou=val_metrics["val_iou"],
                    val_dice=val_metrics["val_dice"],
                    val_imp=val_metrics.get("val_imp_mean", 0),
                    val_cover=val_metrics.get("val_coverage", 0),
                    lr=runner.optimizer.param_groups[0]["lr"] if runner.optimizer else 0,
                )
                log.log_memory(step)
            return log_cb

        runner.on_eval(make_log_cb(seed, log))

        def make_best_cb(seed, log, backbone, decoder, spm):
            def best_cb(step, best_metric):
                save_checkpoint(
                    backbone, decoder, spm,
                    log.run_dir / f"best_s{seed}.pt", best_metric
                )
                log.log_best(best_dice=best_metric, best_step=step)
            return best_cb

        runner.on_best(make_best_cb(seed, log, backbone, decoder, spm))

        # Run training
        gs = 0
        best_dice = 0.0
        seed_base = seed * 1000
        shot = args.n_shot if args.n_shot > 0 else (len(full_ds) - 1)

        # Manual training loop (keeps backward compat with existing logging granularity)
        decoder.train()
        if spm is not None:
            spm.train()

        log_interval = max(1, args.max_steps // 20)

        for epoch in range(args.epochs):
            if args.episodic:
                from adatile.evaluation import split_support_query
                s_idx, q_idx = split_support_query(full_ds, shot, seed_base + epoch)
                q_subset = torch.utils.data.Subset(full_ds, q_idx)
                dl = torch.utils.data.DataLoader(q_subset, batch_size=1, shuffle=True, num_workers=0)
            else:
                dl = torch.utils.data.DataLoader(full_ds, batch_size=batch_sz, shuffle=True, num_workers=0)

            for batch in dl:
                img = batch["images"].to(device)
                gt = batch["masks"].to(device)
                feats = backbone(img)
                lgs = decoder(features=feats)

                imp = spm(feats) if spm is not None else None

                loss, tm = loss_fn(lgs, gt, imp)
                loss.backward()
                params = collect_params(backbone, decoder, spm, loss_fn)
                nn.utils.clip_grad_norm_(params, 1.0)
                # Create optimizer/scheduler if not exists (first step)
                if gs == 0:
                    opt = runner.optimizer
                    sch = runner.scheduler
                    stopper = runner.stopper
                opt.step()
                sch.step()
                opt.zero_grad()
                gs += 1

                if gs % log_interval == 0:
                    decoder.eval()
                    if spm is not None:
                        spm.eval()

                    if args.use_planner:
                        sr = sparse_eval(backbone, decoder, spm, vl, device, args)
                        val_iou = sr.get("sparse_iou", sr.get("full_iou", 0))
                        val_dice = sr.get("sparse_dice", 0)
                        val_metrics = {
                            "val_iou": val_iou, "val_dice": val_dice,
                            "val_imp": sr.get("imp_mean", 0),
                            "val_cover": sr.get("coverage", 0),
                            "sparse_iou": val_iou, "sparse_dice": val_dice,
                            "keep_ratio": sr.get("keep_ratio", 0),
                        }
                    else:
                        with torch.no_grad():
                            vb = next(iter(vl))
                            vi = vb["images"].to(device)
                            vg = vb["masks"].to(device)
                            _, vm = loss_fn(decoder(features=backbone(vi)), vg, None)
                        val_iou, val_dice = vm["iou"], vm["dice"]
                        val_metrics = {"val_iou": val_iou, "val_dice": val_dice}

                    # Enhanced paper-grade logging
                    plog.log_step(
                        step=gs,
                        loss=loss,
                        metrics=tm,
                        importance=imp,
                        val_metrics=val_metrics,
                        lr=opt.param_groups[0]["lr"],
                    )

                    log.log_memory(gs)
                    if val_dice > best_dice:
                        best_dice = val_dice
                        save_checkpoint(backbone, decoder, spm, log.run_dir / f"best_s{seed}.pt", best_dice)
                        log.log_best(best_dice=best_dice, best_step=gs)

                    imp_str = f" imp={imp.mean():.3f}" if imp is not None else ""
                    cov_str = f" cover={tm.get('coverage', 0):.2%}" if imp is not None else ""
                    print(f"  [{gs}] loss={loss.item():.3f} dice={tm['dice']:.3f}{imp_str}{cov_str}  val_dice={val_dice:.3f}")

                    if stopper.step(val_dice):
                        break
                if gs >= args.max_steps:
                    break
            if gs >= args.max_steps or stopper.should_stop:
                break

        # Final evaluation
        fiou, fdice = 0, 0
        if args.use_planner:
            sr = sparse_eval(backbone, decoder, spm, vl, device, args)
            fiou, fdice = sr.get("full_iou", 0), sr.get("sparse_dice", 0)
        else:
            with torch.no_grad():
                decoder.eval()
                if spm is not None:
                    spm.eval()
                ious, dices = [], []
                for vb in vl:
                    feats = backbone(vb["images"].to(device))
                    _, vm = UnifiedLoss(False, args.num_classes)(
                        decoder(features=feats), vb["masks"].to(device), None
                    )
                    ious.append(vm["iou"])
                    dices.append(vm["dice"])
                fiou, fdice = np.mean(ious), np.mean(dices)
                decoder.train()
                if spm is not None:
                    spm.train()

        log.log(args.max_steps + 1, iou=fiou, dice=fdice)
        print(f"  => Seed {seed}: IoU={fiou:.4f} Dice={fdice:.4f}")

        # Record stop reason for this seed
        stop_reason = (
            f"Early stopping at step {gs} (patience=15)"
            if stopper.should_stop
            else f"Max steps reached ({args.max_steps})"
        )
        plog.anomaly.record_stop_reason(f"[Seed {seed}] {stop_reason}")

        if best_dice > _best_overall_dice:
            _best_overall_dice = best_dice
            _best_overall_step = gs

        del backbone, decoder
        if spm is not None:
            del spm
        torch.cuda.empty_cache()

    # Paper-grade finish: auto-visualization + experiment_summary.md
    plog.finish(
        best_dice=_best_overall_dice,
        best_step=_best_overall_step,
        n_params=_n_params_total,
        checkpoint_path=str(log.run_dir / f"best_s{args.seeds-1}.pt"),
    )


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 300
        a.image_size = 640
    main()
