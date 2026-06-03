#!/usr/bin/env python
"""Evaluation entry point for AdaTile-FastSAM.

Usage:
    python tools/eval.py --config configs/isaid.yaml --checkpoint checkpoints/best_model.pt
    python tools/eval.py --config configs/fewshot/1shot.py --checkpoint checkpoints/best_model.pt --fewshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaTile-FastSAM Evaluation")
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Config path (.yaml) or Python module path.",
    )
    parser.add_argument(
        "--checkpoint", "-w", type=str, required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="eval_results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--fewshot", action="store_true",
        help="Run few-shot evaluation.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualization outputs.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from adatile.config import Config

    # Load config
    if args.config.endswith(".yaml"):
        cfg = Config.from_yaml(args.config)
    else:
        import importlib
        parts = args.config.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        cfg = getattr(module, parts[1])()

    # Build model and load weights
    from adatile.modeling import build_adatile_fastsam
    model = build_adatile_fastsam(cfg)
    model.to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Step: {checkpoint.get('step', 'N/A')}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    # Build dataset
    from adatile.datasets import CocoDataset
    from torch.utils.data import DataLoader

    dataset = CocoDataset(
        root_dir=cfg.data.root_dir,
        split="val",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: {
            "images": torch.stack([
                torch.from_numpy(b["image"]).permute(2, 0, 1).float() / 255.0
                for b in batch
            ]),
            "annotations": [b["annotations"] for b in batch],
            "image_ids": [b["image_id"] for b in batch],
        },
    )

    # Run evaluation
    from adatile.evaluation import COCOEvaluator
    evaluator = COCOEvaluator(cfg)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(args.device)
            output, aux = model(images)
            evaluator.process(output, batch)

    metrics = evaluator.evaluate()
    print("\nEvaluation Results:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save predictions
    import json
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
