"""Model builders — construct AdaTile-FastSAM pipelines.

The old Config-based pipeline builder (AdaTileFastSAM class) has been
removed. The active pipeline is built via:

    from adatile.engine import build_components
    backbone, decoder, spm = build_components(args, device, num_classes)

See tools/train_as_fastsam.py for the canonical training entry point.
"""
