# iSAID-5i 数据集规格 | Dataset Specification

> **最后更新**: 2026-07-03
> **数据路径**: `data/iSAID-5i/` (本地), `/root/autodl-tmp/iSAID-5i/` (云服务器)
> **用途**: Few-Shot Semantic Segmentation (FSS) 基准测试

---

## 1. 数据集概述 | Overview

iSAID-5i 是一个基于 iSAID 大尺寸航空图像数据集构建的 Few-Shot 语义分割基准。
原始 iSAID 包含 15 个目标类别，iSAID-5i 通过 **3-Fold 交叉验证** 将其组织为 Base/Novel 划分。

| 属性 | 值 |
|------|-----|
| 原始大图 | 2,806 张 (train) + 验证 |
| Tile 尺寸 | 256×256 px |
| 前景类别 | 15 类 |
| Fold 数 | 3 (Fold 0/1/2) |
| Base 类/Novel 类 | 10 Base / 5 Novel |
| 数据格式 | `images/` (PNG) + `semantic_png/` (PNG 语义掩码) |

---

## 2. 15 个类别全表 | Complete 15-Class Table

```
ID    类别名 (EN)              类别名 (CN)        类型
──────────────────────────────────────────────────────────
 1    ship                     船                  大目标
 2    storage_tank             储罐                中等目标
 3    baseball_diamond         棒球场              中等目标
 4    tennis_court             网球场              中等目标
 5    basketball_court         篮球场              中等目标
 6    ground_track_field       田径场              中等目标
 7    bridge                   桥梁                线状/小目标
 8    large_vehicle            大型车辆            小目标
 9    small_vehicle            小型车辆            极小目标
10    helicopter               直升机              极小目标
11    swimming_pool            游泳池              小目标
12    roundabout               环岛                中等目标
13    soccer_ball_field        足球场              中等目标
14    plane                    飞机                小目标
15    harbor                   港口                大目标
```

---

## 3. Fold 划分 | Fold Splits

### 3.1 Fold 0

```
Base 类 (10) — 训练目标                   Novel 类 (5) — 像素设为 ignore (255)
─────────────────────────────────────────────────────────────────────────────
 1  ship                      船             9  small_vehicle        小型车辆
 2  storage_tank              储罐           11  swimming_pool        游泳池
 3  baseball_diamond          棒球场         12  roundabout           环岛
 4  tennis_court              网球场          5  basketball_court     篮球场
 6  ground_track_field        田径场         15  harbor               港口
 7  bridge                    桥梁
 8  large_vehicle             大型车辆
10  helicopter                直升机
13  soccer_ball_field         足球场
14  plane                     飞机
```

**每个 Base 类的 tile 数量**:

| ID | 类别 | Train Tiles | Val Tiles | 像素占比 | 数据等级 |
|:--:|------|:-----------:|:---------:|:--------:|:--------:|
| 1 | ship | 4,335 | 1,592 | 36.14% | Dominant |
| 4 | tennis_court | 2,592 | 813 | 35.92% | Dominant |
| 2 | storage_tank | 961 | 280 | 12.69% | Moderate |
| 3 | baseball_diamond | 542 | 240 | 7.29% | Moderate |
| 13 | soccer_ball_field | 304 | 132 | 3.60% | Low |
| 6 | ground_track_field | 271 | 128 | 3.22% | Low |
| 8 | large_vehicle | 384 | 148 | 0.49% | Rare |
| 7 | bridge | 72 | 13 | 0.57% | Rare |
| 14 | plane | 14 | 6 | 0.08% | Very Rare |
| 10 | helicopter | 2 | 0 | 0.00% | Extremely Rare |
| — | *(Novel 类不计入训练)* | | | | |
| 9 | small_vehicle | 1,615 | 612 | — | Novel (ignored) |
| 15 | harbor | 3,234 | 1,207 | — | Novel (ignored) |
| 11 | swimming_pool | 131 | 31 | — | Novel (ignored) |
| 5 | basketball_court | 660 | 183 | — | Novel (ignored) |
| 12 | roundabout | 13 | 3 | — | Novel (ignored) |

> **总计**: Train 9,090 tiles (含 FG 的 tile: 8,778), Val 3,108 tiles (含 FG 的 tile: 3,010)

### 3.2 Fold 1

```
Base 类 (10) — 训练目标                   Novel 类 (5) — 像素设为 ignore (255)
─────────────────────────────────────────────────────────────────────────────
 1  ship                      船            14  plane                 飞机
 2  storage_tank              储罐           8  large_vehicle         大型车辆
 3  baseball_diamond          棒球场          7  bridge                桥梁
 5  basketball_court          篮球场          6  ground_track_field    田径场
 9  small_vehicle             小型车辆        4  tennis_court          网球场
10  helicopter                直升机
11  swimming_pool             游泳池
12  roundabout                环岛
13  soccer_ball_field         足球场
15  harbor                    港口
```

**每个 Base 类的 tile 数量**:

| ID | 类别 | Train Tiles | Val Tiles | 数据等级 |
|:--:|------|:-----------:|:---------:|:--------:|
| 9 | small_vehicle | 5,302 | 1,814 | Dominant |
| 8 | large_vehicle | 3,931 | 1,190 | Dominant |
| 1 | ship | 907 | 333 | Moderate |
| 6 | ground_track_field | 1,478 | 547 | Moderate (Novel↓) |
| 13 | soccer_ball_field | 1,012 | 400 | Moderate |
| 15 | harbor | 648 | 266 | Low |
| 14 | plane | 436 | 208 | Low (Novel↓) |
| 4 | tennis_court | 615 | 227 | Low (Novel↓) |
| 3 | baseball_diamond | 160 | 72 | Rare |
| 2 | storage_tank | 221 | 90 | Rare |
| 7 | bridge | 246 | 96 | Rare (Novel↓) |
| 10 | helicopter | 78 | 10 | Very Rare |
| 12 | roundabout | 147 | 42 | Very Rare |
| 5 | basketball_court | 204 | 71 | Very Rare |
| 11 | swimming_pool | 82 | 33 | Very Rare |

> **总计**: Train 11,035 tiles, Val 2,826 tiles (含 FG)

### 3.3 Fold 2

```
Base 类 (10) — 训练目标                   Novel 类 (5) — 像素设为 ignore (255)
─────────────────────────────────────────────────────────────────────────────
 4  tennis_court              网球场          1  ship                  船
 5  basketball_court          篮球场          2  storage_tank          储罐
 6  ground_track_field        田径场         10  helicopter            直升机
 7  bridge                    桥梁           13  soccer_ball_field     足球场
 8  large_vehicle             大型车辆        3  baseball_diamond      棒球场
 9  small_vehicle             小型车辆
11  swimming_pool             游泳池
12  roundabout                环岛
14  plane                     飞机
15  harbor                    港口
```

**每个 Base 类的 tile 数量**:

| ID | 类别 | Train Tiles | Val Tiles | 数据等级 |
|:--:|------|:-----------:|:---------:|:--------:|
| 9 | small_vehicle | 1,379 | 603 | Dominant |
| 15 | harbor | 3,624 | 1,432 | Dominant |
| 1 | ship | 3,254 | 1,203 | Dominant (Novel↓) |
| 13 | soccer_ball_field | 2,131 | 738 | Dominant (Novel↓) |
| 14 | plane | 1,890 | 1,000 | Moderate |
| 6 | ground_track_field | 873 | 291 | Moderate |
| 3 | baseball_diamond | 58 | 32 | Rare (Novel↓) |
| 8 | large_vehicle | 386 | 226 | Rare |
| 11 | swimming_pool | 243 | 74 | Rare |
| 4 | tennis_court | 226 | 85 | Rare |
| 12 | roundabout | 219 | 54 | Rare |
| 5 | basketball_court | 97 | 47 | Very Rare |
| 2 | storage_tank | 35 | 22 | Very Rare (Novel↓) |
| 7 | bridge | 44 | 9 | Very Rare |
| 10 | helicopter | 14 | 6 | Very Rare (Novel↓) |

> **总计**: Train 8,107 tiles, Val 3,273 tiles (含 FG)

---

## 4. Base/Novel 快速对照 | Quick Reference

```
Class                Fold 0        Fold 1        Fold 2
─────────────────────────────────────────────────────────
 1  ship              BASE          BASE          NOVEL
 2  storage_tank      BASE          BASE          NOVEL
 3  baseball_diamond  BASE          BASE          NOVEL
 4  tennis_court      BASE          NOVEL         BASE
 5  basketball_court  NOVEL         BASE          BASE
 6  ground_track      BASE          NOVEL         BASE
 7  bridge            BASE          NOVEL         BASE
 8  large_vehicle     BASE          NOVEL         BASE
 9  small_vehicle     NOVEL         BASE          BASE
10  helicopter        BASE          BASE          NOVEL
11  swimming_pool     NOVEL         BASE          BASE
12  roundabout        NOVEL         BASE          BASE
13  soccer_ball_field BASE          BASE          NOVEL
14  plane             BASE          NOVEL         BASE
15  harbor            NOVEL         BASE          BASE
```

**每类在哪个 Fold 是 Novel**:

| Novel 在 | 类 |
|:--------:|----|
| Fold 0 | basketball_court(5), small_vehicle(9), swimming_pool(11), roundabout(12), harbor(15) |
| Fold 1 | tennis_court(4), ground_track(6), bridge(7), large_vehicle(8), plane(14) |
| Fold 2 | ship(1), storage_tank(2), baseball_diamond(3), helicopter(10), soccer(13) |

**每类在哪个 Fold 是 Base**:

| Base 在 | 类 |
|:-------:|----|
| Fold 0 | ship, storage_tank, baseball_diamond, tennis_court, ground_track, bridge, large_vehicle, helicopter, soccer, plane |
| Fold 1 | ship, storage_tank, baseball_diamond, basketball_court, small_vehicle, helicopter, swimming_pool, roundabout, soccer, harbor |
| Fold 2 | tennis_court, basketball_court, ground_track, bridge, large_vehicle, small_vehicle, swimming_pool, roundabout, plane, harbor |

---

## 5. 数据路径结构 | Directory Structure

```
iSAID-5i/
├── train/
│   ├── images/                  # 256×256 PNG 图像
│   ├── semantic_png/            # 语义掩码 (category_id 像素值)
│   └── train_list/
│       ├── split0_train.txt     # Fold 0 训练 tile 列表 (9,090)
│       ├── split1_train.txt     # Fold 1 训练 tile 列表 (11,035)
│       └── split2_train.txt     # Fold 2 训练 tile 列表 (8,107)
├── val/
│   ├── images/                  # 256×256 PNG 图像
│   ├── semantic_png/            # 语义掩码
│   └── val_list/
│       ├── split0_val.txt       # Fold 0 验证 tile 列表 (3,108)
│       ├── split1_val.txt       # Fold 1 验证 tile 列表
│       └── split2_val.txt       # Fold 2 验证 tile 列表
└── label.xlsx                   # 类别标签定义
```

---

## 6. 与当前实验的关系 | Relation to Experiment A

Experiment A 使用 **Fold 0, Base-only** 模式进行全监督训练：

- **训练目标**: Fold 0 的 10 个 Base 类
- **忽略**: Fold 0 的 5 个 Novel 类 → 像素映射为 255
- **为什么选 Fold 0**: 参考表 (REFERENCE_TABLE) 数据匹配 Fold 0 的 split，便于与其他论文对比

**论文中常见术语对应**:
- `Base classes` = `seen classes` = `源域类别`
- `Novel classes` = `unseen classes` = `目标域类别`
- `K-shot` = 每个 Novel 类使用 K 个标注样本做 few-shot 学习

---

## 7. 备注 | Notes

1. **3-Fold 交叉验证**: 每个 fold 不仅 Base/Novel 类划分不同，**train/val tile 划分也不同**。同一张原始大图切出的 tile 被分配到不同 fold 的 train/val 中，所以各 fold 的同类别 tile 数量不相等。同一类在 Fold 0 和 Fold 2 的 tile 数量不相等（如 harbor: Fold 0=3,234, Fold 2=3,624），因为使用的是 `split0_train.txt` 和 `split2_train.txt` 这两个不同的 tile 子集。这是 iSAID-5i 的标准设计。
2. **参考表来源**: `REFERENCE_TABLE` 中的 train/val 计数取自各 fold 下该类的 Base fold 数据（即 class 1-5 取 Fold 0，class 6-10 取 Fold 1，class 11-15 取 Fold 2），并非单一 fold 的统计。
3. **10 Base vs 5 Base**: 部分论文使用 5 Base / 10 Novel 的划分（更严格），本代码库使用标准 iSAID-5i 的 10 Base / 5 Novel 划分。如需切换到 5 Base 模式，需修改 `ISAID5I_FOLDS` 或创建新的 fold 定义。
4. **数据稀缺**: helicopter(10) 在所有 fold 中都是极稀缺类，Fold 0 train 仅 2 tiles，1.4K 像素。在 256² (16×16 feature) 下基本不可学习。
