# NEU_Seg — Comprehensive Dataset Analysis Report

**Generated:** 2026-07-15 22:09:41

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | NEU_Seg |
| Total Images | 4470 |
| Total Masks | 4470 |
| Image Size | 200x200 px |
| Channels | 3 (RGB) |
| Mask Encoding | Original: uint8 {0,1,2,3} (0=BG, 1=Inclusion, 2=Patch, 3=Scratch); Binary: FG>0→1 |
| train Split | 3630 images, 3630 masks, 200x200 |
| test Split | 840 images, 840 masks, 200x200 |

## 2. Class Statistics

| Class | Encoding | Images | Pixel % |
|-------|----------|--------|---------|
| background | 0 | 4170 | 87.9% |
| Inclusion | 1 | 1824 | 2.2% |
| Patch | 2 | 1808 | 7.3% |
| Scratch | 3 | 1762 | 2.6% |

- **Class Balance:** Imbalanced (max/min FG class ratio 3.3x)
- **Global FG Ratio:** 12.1% pixels are defect (non-BG)

## 3. Image Quality Analysis

### 3.1 Brightness & Contrast

- **Brightness:** mean=112.6/255, std=37.4
- **Dynamic Range:** min-max spread
- **Overexposed:** 52.9% (definition: max_pixel > 250/255)
- **Underexposed:** 14.1% (definition: min_pixel < 5/255)

### 3.2 Image Sharpness

- **Blur Detection:** Laplacian Variance < 100.0
- **Blurry Images:** 2271/4470 (50.81%)
- **Laplacian Var:** mean=439.3 (±587.5)

## 4. Annotation Quality Analysis

- **FG Ratio (all):** mean=0.1214, median=0.0963
- **FG Ratio (non-empty):** see Defect Size Distribution (M06)
- **Components/Image:** mean=3.0, max=21
- **Empty Masks (no FG):** 0.0%
- **Object Area:** mean=1629 px, median=670 px
- **Object Area Range:** 1–27832 px

## 5. Defect Size Distribution

- **Total Objects:** 13320
- **Small (<1024px):** 7965 (59.8%)
- **Medium (1024–9216px):** 5056 (38.0%)
- **Large (>9216px):** 299 (2.2%)

**Area Percentiles:** p25=199px, p50=670px, p75=2030px, p95=6438px, p99=11830px

### 5.1 Defect Size by FG Ratio

| Category | Count | % |
|----------|-------|---|
| Tiny (<0.5%) | 42 | 0.9% |
| Small (0.5-2%) | 374 | 8.4% |
| Medium (2-5%) | 859 | 19.2% |
| Large (>5%) | 3195 | 71.5% |

## 6. Distribution Shift (Train vs Test)

| Metric | KS D | p-value | Significant? |
|--------|------|---------|--------------|
| FG Ratio | 0.0514 | 0.0524 | No (p>=0.05) |
| Brightness | 0.0685 | 0.0031 | Yes (p<0.05) |
| Laplacian Var | 0.1112 | 0.0000 | Yes (p<0.05) |

**Interpretation:** Significant shifts in brightness (p=0.0031) and blur (p=0.0000) indicate train/test domain gap. FG ratio distribution (p=0.0524) is consistent between splits.

## 7. Few-shot Adaptation Analysis

- **FG Samples Available:** 3630/3630
- **Support Sampling Stability (PS):** 0.039421
  - Definition: PS = std(bootstrap mean of FG ratio from K=5 random support), 100 iterations.
  - Lower PS → support set choice has less impact on FG ratio estimation.
  - Note: This measures pixel-level FG ratio stability, NOT feature-space prototype similarity.

**Episode definition:** For K-shot, a support set of K FG images is drawn without replacement. Episodes = floor(FG_samples / K). Coverage = fraction of FG samples used across all episodes.

| K | FG Samples | Possible Episodes | Support Coverage |
|---|-----------|-------------------|-----------------|
| K=1 | 3630 | 3630 | 100.0% |
| K=5 | 3630 | 726 | 100.0% |
| K=10 | 3630 | 363 | 100.0% |

## 8. Texture & Shape Characteristics

### 8.1 GLCM Texture

| Property | Mean ± Std | Description |
|----------|-----------|-------------|
| Contrast | 6.918 ± 11.325 | Local intensity variation |
| Energy | 0.1625 ± 0.0625 | Texture uniformity (higher=more uniform) |
| Homogeneity | 0.615 ± 0.187 | Similarity of neighboring pixels |
| Correlation | 0.939 ± 0.033 | Linear dependency of neighboring pixels |

### 8.2 Shape Metrics

| Metric | Mean | Median | Description |
|--------|------|--------|-------------|
| Circularity | 0.384 | 0.335 | 4πA/P² (1=circle) |
| Eccentricity | 0.908 | 0.983 | 0=circle, 1=line |
| Convexity | 0.919 | 0.976 | Area/HullArea (1=convex) |
| Elongation | 5.8 | 4.2 | Max(W,H)/Min(W,H) |

## 9. Data Quality Issues

- **Quality Errors:** 0
- **Quality Warnings:** 0
- **Duplicate Images:** 4 (0.09%)
- **Duplicate Masks:** 0
- **Spatial Bias:** uniform (center mean=(0.499,0.499))

## 10. Recommended Data Augmentation Strategies

| Augmentation | Rationale |
|-------------|-----------|
| RandomFlip (H+V), RandomRotate90 | Objects have no canonical orientation |
| RandomBrightnessContrast | Significant brightness distribution shift between train/test (KS p<0.01) |
| GaussianNoise | Improve robustness to blur variation (50.8% blurry, KS p<0.001) |
| FG-Weighted Sampling | FG pixels=12.1% of total; weighted sampling reduces background dominance |
| RandomCrop (larger context) | 200x200 is small; expanding receptive field helps capture context |

## 11. Task Suitability Assessment

| Task | Suitable? | Notes |
|------|-----------|-------|
| Semantic Segmentation | ✅ Yes | Binary FG/BG masks, well-defined |
| Instance Segmentation | ⚠️ Partial | Connected components separable but no instance IDs |
| Few-shot Segmentation | ✅ Yes | 3630 FG support candidates, K=1 viable |
| Anomaly Detection | ✅ Yes | Binary normal/defect framing |
| Foundation Model Fine-tuning | ✅ Yes | Compatible with SAM/FastSAM binary mask format |

## 12. Key Metrics Summary

| Metric | Value | Paper-Ready? |
|--------|-------|-------------|
| Samples (Train/Test) | 3630/840 | ✅ |
| Image Size | 200x200 | ✅ |
| FG/BG Pixel Ratio | 1:7.2 | ✅ |
| Object Area Median | 670 px | ✅ |
| Object Size Split | S=59.8%/M=38.0%/L=2.2% | ✅ |
| Brightness Shift | KS p=0.0031 | ✅ |
| Blur Shift | KS p=0.0000 | ✅ |
| Prototype Stability | 0.039421 | ✅ |
| Duplicate Rate | 0.09% | ✅ |
| Spatial Bias | uniform | ✅ |