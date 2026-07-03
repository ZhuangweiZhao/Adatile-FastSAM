# iSAID Instance Few-Shot Split 数据集规格 | v3 Dataset Specification

> **最后更新**: 2026-07-03
> **数据路径**: `data/iSAID_instance_fewshot/` (本地), `/root/autodl-tmp/iSAID_instance_fewshot/` (云)
> **用途**: High-Resolution Few-Shot Instance Segmentation (v3 协议)
> **生成脚本**: `tools/data/prep_isaid_instance.py`

---

## 1. 数据集概述 | Overview

基于 iSAID COCO 全图切分的 896×896 tile 实例分割数据集。弃用 iSAID-5i (FSS protocol)，自定义 Base/Novel 3-Fold 划分。

| 属性 | 值 |
|------|-----|
| 原始大图 | 1,411 (train) + 458 (val) |
| Tile 尺寸 | 896×896 px |
| 滑动步长 | 512 px |
| 前景类别 | 15 类 |
| Fold 数 | 3 (Fold 0/1/2) |
| Base 类/Novel 类 | 10 Base / 5 Novel |
| 标注格式 | COCO JSON (polygon segmentation, per-instance) |

---

## 2. 15 个类别 | 15 Categories

使用标准 ISAID_CATEGORIES 编码（来自 `adatile.utils.label_mapping`）：

```
ID    类别名 (EN)              类别名 (CN)        类型
──────────────────────────────────────────────────────────
 1    small_vehicle             小型车辆            极小目标
 2    large_vehicle             大型车辆            小目标
 3    plane                     飞机                小目标
 4    storage_tank              储罐                中等目标
 5    ship                      船                  大目标
 6    harbor                    港口                大目标
 7    ground_track_field        田径场              大目标
 8    soccer_ball_field         足球场              大目标
 9    tennis_court              网球场              中等目标
10    swimming_pool             游泳池              小目标
11    baseball_diamond          棒球场              中等目标
12    basketball_court          篮球场              中等目标
13    bridge                    桥梁                线状/小目标
14    helicopter                直升机              极小目标
15    roundabout                环岛                中等目标
```

---

## 3. 数据量统计 | Dataset Statistics

### 3.1 总体

| Split | Tiles | Instances | Empty Tiles | Empty% |
|-------|------:|----------:|:-----------:|:------:|
| Train | 24,113 | 666,628 | 7,988 | 33.1% |
| Val | 8,331 | 225,619 | 3,028 | 36.3% |

### 3.2 Per-Class (Train Split)

```
Class                    ID    Instances   Tiles    Mean Area  等级
──────────────────────────────────────────────────────────────────────
small_vehicle             1     456,861     8,394     139 px²   Dominant ★
large_vehicle             2      68,154     5,561     755 px²   Dominant ★
plane                     3      21,944     3,534   2,752 px²   High
storage_tank              4      14,856     1,236   1,941 px²   High
ship                      5      71,201     3,368     964 px²   Dominant ★
harbor                    6      12,331     2,190   3,401 px²   High
ground_track_field        7       1,101       913  61,274 px²   Low (大目标)
soccer_ball_field         8       1,442       919  60,672 px²   Low (大目标)
tennis_court              9       3,754       707   8,780 px²   Moderate
swimming_pool            10       4,544     1,034   1,238 px²   Moderate
baseball_diamond         11       1,007       488   7,954 px²   Low
basketball_court         12       1,236       361  11,971 px²   Low
bridge                   13       5,308     2,447   1,404 px²   Moderate
helicopter               14       1,899       204   1,047 px²   Very Low ⚠
roundabout               15         990       773   7,548 px²   Low
──────────────────────────────────────────────────────────────────────
Total                           666,628    24,113
```

### 3.3 Per-Class (Val Split)

| Class | ID | Instances | Tiles | Mean Area |
|-------|:--:|----------:|------:|----------:|
| small_vehicle | 1 | 160,651 | 2,948 | 117 px² |
| large_vehicle | 2 | 26,543 | 2,219 | 960 px² |
| plane | 3 | 8,313 | 1,422 | 3,488 px² |
| storage_tank | 4 | 3,378 | 461 | 2,416 px² |
| ship | 5 | 20,111 | 1,314 | 1,119 px² |
| harbor | 6 | 3,465 | 777 | 4,010 px² |
| ground_track_field | 7 | 344 | 278 | 58,874 px² |
| soccer_ball_field | 8 | 536 | 342 | 61,888 px² |
| tennis_court | 9 | 1,465 | 267 | 7,989 px² |
| swimming_pool | 10 | 1,223 | 262 | 849 px² |
| baseball_diamond | 11 | 178 | 115 | 7,931 px² |
| basketball_court | 12 | 451 | 130 | 10,369 px² |
| bridge | 13 | 1,201 | 430 | 1,747 px² |
| helicopter | 14 | 317 | 75 | 929 px² |
| roundabout | 15 | 400 | 258 | 5,571 px² |

---

## 4. 3-Fold Base/Novel 划分 | Fold Splits

### 4.1 Fold 0

```
Base (10):                          Novel (5):
─────────────────────────────────────────────────────────
 2  large_vehicle       大型车辆      1  small_vehicle       小型车辆
 3  plane               飞机           6  harbor              港口
 4  storage_tank        储罐          10  swimming_pool       游泳池
 5  ship                船            12  basketball_court    篮球场
 7  ground_track_field  田径场        15  roundabout          环岛
 8  soccer_ball_field   足球场
 9  tennis_court        网球场
11  baseball_diamond    棒球场
13  bridge              桥梁
14  helicopter          直升机
```

**Base 类资源 (Train)**:

| ID | 类 | Instances | Tiles |
|:--:|-----|----------:|------:|
| 2 | large_vehicle | 68,154 | 5,561 |
| 3 | plane | 21,944 | 3,534 |
| 4 | storage_tank | 14,856 | 1,236 |
| 5 | ship | 71,201 | 3,368 |
| 7 | ground_track_field | 1,101 | 913 |
| 8 | soccer_ball_field | 1,442 | 919 |
| 9 | tennis_court | 3,754 | 707 |
| 11 | baseball_diamond | 1,007 | 488 |
| 13 | bridge | 5,308 | 2,447 |
| 14 | helicopter | 1,899 | 204 |
| **Total** | | **190,666** | — |

**Novel 类资源 (Train)**:

| ID | 类 | Instances | Tiles |
|:--:|-----|----------:|------:|
| 1 | small_vehicle | 456,861 | 8,394 |
| 6 | harbor | 12,331 | 2,190 |
| 10 | swimming_pool | 4,544 | 1,034 |
| 12 | basketball_court | 1,236 | 361 |
| 15 | roundabout | 990 | 773 |
| **Total** | | **475,962** | — |

### 4.2 Fold 1

```
Base (10):                          Novel (5):
─────────────────────────────────────────────────────────
 1  small_vehicle       小型车辆      2  large_vehicle       大型车辆
 4  storage_tank        储罐           3  plane               飞机
 5  ship                船             7  ground_track_field  田径场
 6  harbor              港口           9  tennis_court        网球场
 8  soccer_ball_field   足球场        13  bridge              桥梁
10  swimming_pool       游泳池
11  baseball_diamond    棒球场
12  basketball_court    篮球场
14  helicopter          直升机
15  roundabout          环岛
```

### 4.3 Fold 2

```
Base (10):                          Novel (5):
─────────────────────────────────────────────────────────
 1  small_vehicle       小型车辆      4  storage_tank        储罐
 2  large_vehicle       大型车辆      5  ship                船
 3  plane               飞机           8  soccer_ball_field   足球场
 6  harbor              港口          11  baseball_diamond    棒球场
 7  ground_track_field  田径场        14  helicopter          直升机
 9  tennis_court        网球场
10  swimming_pool       游泳池
12  basketball_court    篮球场
13  bridge              桥梁
15  roundabout          环岛
```

### 4.4 快速对照

```
Class                Fold 0        Fold 1        Fold 2
─────────────────────────────────────────────────────────
 1  small_vehicle     NOVEL         BASE          BASE
 2  large_vehicle     BASE          NOVEL         BASE
 3  plane             BASE          NOVEL         BASE
 4  storage_tank      BASE          BASE          NOVEL
 5  ship              BASE          BASE          NOVEL
 6  harbor            NOVEL         BASE          BASE
 7  ground_track      BASE          NOVEL         BASE
 8  soccer_ball_field BASE          BASE          NOVEL
 9  tennis_court      BASE          NOVEL         BASE
10  swimming_pool     NOVEL         BASE          BASE
11  baseball_diamond  BASE          BASE          NOVEL
12  basketball_court  NOVEL         BASE          BASE
13  bridge            BASE          NOVEL         BASE
14  helicopter        BASE          BASE          NOVEL
15  roundabout        NOVEL         BASE          BASE
```

---

## 5. 数据路径结构 | Directory Structure

```
iSAID_instance_fewshot/
├── images/
│   ├── train/                     # 24,113 × 896×896 PNG tiles
│   └── val/                       #  8,331 × 896×896 PNG tiles
├── annotations/
│   ├── instances_train.json       # COCO JSON (666,628 instances)
│   └── instances_val.json         # COCO JSON (225,619 instances)
├── folds/
│   ├── fold_0.json                # Base/Novel class ID lists
│   ├── fold_1.json
│   └── fold_2.json
└── stats/
    └── class_distribution.json    # Per-class instance/tile/pixel stats
```

---

## 6. 数据等级定义 | Data Tier Definitions

| 等级 | Instances Threshold | 类 | 说明 |
|:----:|:--------------------|-----|------|
| **Dominant** | >50,000 | small_vehicle, ship, large_vehicle | 充裕，可做各类消融 |
| **High** | 10,000–50,000 | plane, storage_tank, harbor | 充足，正常训练 |
| **Moderate** | 3,000–10,000 | tennis_court, swimming_pool, bridge | 够用，K-shot 可行 |
| **Low** | 500–3,000 | ground_track, soccer, baseball, basketball, roundabout | 勉强，需注意过拟合 |
| **Very Low** | <500 | helicopter (1,899 实例 / 204 tiles) | ⚠ 极易过拟合，需特殊处理 |

---

## 7. 与旧协议 (iSAID-5i) 对比 | Comparison with iSAID-5i

| 维度 | iSAID-5i (v2) | Instance Few-Shot (v3) |
|------|---------------|------------------------|
| Tile 尺寸 | 256×256 px | **896×896 px** |
| Tile 数量 (train) | 9,090 | **24,113** |
| 标注格式 | Semantic PNG (per-pixel class ID) | **COCO JSON (polygon)** |
| 实例信息 | 丢失 (通过连通分量推断) | **完整 (per-instance polygon)** |
| 协议 | FSS (Episode-based, Support/Query) | **Few-Shot Fine-tuning (Base→Novel)** |
| 主评价 | mIoU | **COCO AP** |
| 生成脚本 | (外部来源) | `tools/data/prep_isaid_instance.py` |

---

## 8. 关键注意事项 | Key Notes

1. **helicopter 是极大挑战**: 仅 1,899 实例 / 204 tiles。对应的 896² tile 中，直升机可能仅占几个像素。Fold 2 中它还是 Novel 类，K-shot 设置下尤其困难。

2. **small_vehicle 主导数据集**: 456,861 实例占总数 68%。但 mean_area=139px² (极小的目标)，896² 下单个 small_vehicle 非常小，需要高分辨率特征 (P3 stride-8)。

3. **Empty tile 率 33-36%**: 与 B-00 空间稀疏性发现一致 (所有尺度都稀疏)。这些 empty tiles 可以作为 BG 负样本。

4. **3-Fold 都可用**: 每个 Fold 都有不同的 Base/Novel 组合。Fold 0 中 Novel 类数量多 (small_vehicle=456K)，Fold 2 中 Novel 类更难 (helicopter=1,899)。

5. **类别 ID 体系**: 使用标准 ISAID_CATEGORIES (1-15)，与 `adatile.utils.label_mapping.ISAID_CATEGORIES` 一致。注意与 iSAID-5i 的 ISAID5I_CATEGORIES 不同。

---

*This document supersedes `docs/isaid5i_dataset_spec.md` for all v3 experiments.*
