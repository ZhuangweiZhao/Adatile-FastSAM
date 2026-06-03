"""AdaTile-FastSAM: Adaptive Sparse Tiling for Fast Instance Segmentation."""

from setuptools import setup, find_packages

setup(
    name="adatile-fastsam",
    version="0.1.0",
    description="Adaptive Sparse Tiling for High-Resolution Few-Shot Instance Segmentation",
    author="",
    packages=find_packages(include=["adatile", "adatile.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pycocotools>=2.0.7",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
        "xformers>=0.0.22",
        "ultralytics>=8.0.0",
        "fvcore>=0.1.5",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "mypy", "ruff", "pre-commit"],
        "viz": ["matplotlib", "seaborn", "wandb"],
    },
)
