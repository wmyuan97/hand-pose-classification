# 手部姿态分类：传统方法

本文说明 `traditional_method/` 当前使用的传统机器视觉方法：通过肤色分割提取手部区域，计算形状特征，再使用 SVM 完成外展（ABD）、内收（ADD）和对掌（OPP）三分类。

## 目录结构

```text
traditional_method/
├── 01segment_hands.py                 肤色分割、形态学处理与手部轮廓筛选
├── 02extract_features.py              从手部掩膜计算特征，生成 features.csv
├── 03evaluate_4fold.py                PCA + RBF-SVM 四折交叉验证
├── 04evaluate_linear_svm.py           Linear SVM 与 PCA + Linear SVM 对比
├── 05compare_pca_variance.py          PCA 方差保留比例对比
├── results/
│   ├── masks/                        肤色分割和形态学处理后的二值掩膜
│   ├── overlays/                     肤色区域叠加图
│   ├── contours/                     保留与剔除轮廓的可视化
│   ├── selected_masks/               供特征提取使用的最终手部掩膜
│   ├── features.csv                  当前 108 个样本的特征和标签
│   └── feature_contribution_108/      已保存的特征分析文件
├── traditional_4fold_results/         已保存的 RBF-SVM 基线及四折名单
│   ├── predictions.xlsx
│   ├── overall_metrics.xlsx
│   ├── group_metrics.xlsx
│   └── confusion_matrices/
├── traditional_pca_4fold_results/     第 03 个脚本的输出
│   ├── predictions.xlsx
│   ├── overall_metrics.xlsx
│   ├── group_metrics.xlsx
│   └── confusion_matrices/
└── linear_svm_4fold_results/          第 04 个脚本的输出
    ├── linear/
    └── linear_pca95/
```

三个四折结果目录都放在 `traditional_method/` 内。`linear/` 和 `linear_pca95/` 分别保存预测表、整体指标、分组指标和混淆矩阵。

## 方法流程

```text
原始图片
  ↓ 等比例缩放到高度 1280 px
YCrCb 与 HSV 肤色条件取交集
  ↓ 形态学开运算、闭运算
筛选并填充手部轮廓
  ↓ 最终二值手部掩膜
提取 13 维形状特征
  ↓ 按拍摄条件进行四折划分
每折：训练集拟合缺失值填补、标准化和可选 PCA → 训练 SVM → 预测测试集
  ↓ 合并四折测试预测
计算整体、各角度和各遮挡条件的评价指标
```

### 1. 手部区域分割

`01segment_hands.py` 读取项目根目录的 `labels.csv`，根据其中的 `path` 加载图片，并依次执行：

1. 将图片等比例缩放到高度 1280 像素。
2. 转换到 YCrCb，保留 `Cr ∈ [133, 173]`、`Cb ∈ [77, 127]` 的候选皮肤像素。
3. 转换到 HSV，要求饱和度 `S ≥ 50`，并将该条件与 YCrCb 掩膜取交集。
4. 使用 5×5 椭圆核做开运算去除小噪点，再使用 11×11 椭圆核做闭运算填补小缺口。
5. 查找外轮廓，剔除面积小于 1000 平方像素的轮廓；保留最大轮廓，以及与最大轮廓包围框距离不超过 50 像素的其他合格轮廓。
6. 将保留的轮廓填充为最终二值掩膜，并保存掩膜、叠加图和轮廓可视化。

### 2. 形状特征提取

`02extract_features.py` 读取 `results/selected_masks/` 中的最终手部掩膜，计算各保留轮廓的总面积、总周长、整体包围框、整体凸包以及二值掩膜的 Hu 矩，保存到 `results/features.csv`。

分类器实际使用以下 13 个特征：

| 特征 | 定义 | 维数 |
|---|---|---:|
| `normalized_area` | 总轮廓面积 / 图片面积 | 1 |
| `normalized_perimeter` | 总轮廓周长 / 图片面积的平方根 | 1 |
| `aspect_ratio` | 整体包围框宽度 / 高度 | 1 |
| `extent` | 总轮廓面积 / 整体包围框面积 | 1 |
| `solidity` | 总轮廓面积 / 整体凸包面积 | 1 |
| `circularity` | 4π × 总轮廓面积 / 总轮廓周长² | 1 |
| `hu_1` 至 `hu_7` | 对 7 个 Hu 矩做带符号的对数变换 | 7 |

特征表还记录文件名、姿态、角度、遮挡、轮廓数量及其他几何统计，供样本对齐和分析使用。

### 3. 特征预处理与分类

每一折都从其训练集拟合完整的处理流水线：

1. 将非有限数值作为缺失值，用训练集各特征的中位数填补。
2. 使用训练集拟合 `StandardScaler`，对特征进行标准化。
3. 使用 PCA 的方案在标准化之后拟合 PCA，按目标累计解释方差确定维数。
4. 训练 SVM，并对该折测试集应用已经拟合的预处理和分类器。

当前脚本中的分类方案为：

| 脚本 | 模型配置 | PCA 设置 |
|---|---|---|
| `03evaluate_4fold.py` | RBF-SVM，`C=1.0`，`gamma="scale"` | 保留 95% 累计解释方差 |
| `04evaluate_linear_svm.py` | Linear SVM，`C=1.0` | 分别使用全部 13 维特征和保留 95% 方差的 PCA 特征 |
| `05compare_pca_variance.py` | Linear SVM，`C=1.0` | 比较 85%、90%、95%、97%、99% 五种方差保留比例 |

PCA 的实际维数由每折训练集决定。整行缺少有效特征的样本在第 03、04 个脚本中标记为未分类，并统计未分类数量；分类指标根据具有有效预测的样本计算。第 05 个脚本要求所有样本至少有一个有效特征。

## 数据划分与评估

当前传统方法的特征表包含 108 个样本。按拍摄条件划分，有 3 种姿态、3 个角度和 3 种遮挡条件：

| 维度 | 取值 |
|---|---|
| 姿态 | 外展 ABD、内收 ADD、对掌 OPP |
| 角度 | 正面 F、45°、90° |
| 遮挡 | 无遮挡 NONE、纸张 PAPER、另一只手 HAND |

三者组合成 27 个条件组，每组包含 4 个样本。第 03 个脚本先按文件名排序，再以 `姿态 + 角度 + 遮挡` 作为联合分层标签，使用 `StratifiedKFold(n_splits=4, shuffle=True, random_state=20261123)` 生成四折：

| 测试折 | 训练样本数 | 测试样本数 |
|---|---:|---:|
| Fold 1 | 81 | 27 |
| Fold 2 | 81 | 27 |
| Fold 3 | 81 | 27 |
| Fold 4 | 81 | 27 |

每折包含每种拍摄条件的 1 个测试样本；每张图片恰好测试 1 次，其余 3 折作为训练样本。训练与测试比例为 75% / 25%，不另设独立验证集。该划分按图片在条件组内分层，不按拍摄者或数据批次隔离。

第 04、05 个脚本读取 `traditional_method/traditional_4fold_results/predictions.xlsx` 中已有的 `filename` 和 `fold` 列，复用同一份测试名单。当前保存的名单已核对，与第 03 个脚本的默认划分一致。

如需让第 03 个脚本复用外部名单，可将其 `SHARED_FOLDS` 设置为 CSV 或 Excel 文件路径。名单需要包含 `filename` 和 `fold`（或 `test_fold`）列，并满足 108 个样本一一对应、折号为 1–4、每种条件在每折恰好出现一次。更换划分时，应同步第 04、05 个脚本读取的名单。

评估时合并四折的测试预测，计算整体准确率、Macro-F1、各类别 Precision / Recall / F1，以及混淆矩阵；再分别按拍摄角度和遮挡条件汇总。

## 安装与脚本运行顺序

以下命令均在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install openpyxl==3.1.5
```

`openpyxl` 用于读取四折名单和写入 Excel 指标文件。

### 使用现有特征继续评估

仓库已保存当前 108 个样本的 `traditional_method/results/features.csv`，以及第 04、05 个脚本所需的 `traditional_method/traditional_4fold_results/predictions.xlsx`。可以按顺序执行：

```bash
python traditional_method/03evaluate_4fold.py
python traditional_method/04evaluate_linear_svm.py
python traditional_method/05compare_pca_variance.py
```

第 03 个脚本写入 `traditional_pca_4fold_results/`；第 04 个脚本写入 `linear_svm_4fold_results/`；第 05 个脚本在终端打印比较表。第 04、05 个脚本使用的基线四折名单保存在 `traditional_4fold_results/`。

### 从原图完整重跑

第 01、02 个脚本需要与当前 108 张图片匹配的完整 `labels.csv`，其中 `path` 必须对应实际图片路径和扩展名。GitHub 根目录的 `labels.csv` 已覆盖全部 108 个样本。`filename` 沿用当前特征表和四折名单中的样本标识，`path` 对应 GitHub 中原图的实际路径；部分图片的 `.jpg` / `.jpeg` 扩展名与样本标识不同。若本地原图扩展名不同，请在本地运行副本中调整 `path`，保留 `filename` 和其他标签。上述“使用现有特征继续评估”的命令直接读取已保存特征。

输入准备完成后，依次执行：

```bash
python traditional_method/01segment_hands.py
python traditional_method/02extract_features.py
python traditional_method/03evaluate_4fold.py
python traditional_method/04evaluate_linear_svm.py
python traditional_method/05compare_pca_variance.py
```

第 01 个脚本生成手部掩膜，第 02 个脚本据此生成特征表。第 04、05 个脚本还需要保留与当前样本及评估划分一致的基线四折名单。

## 实验结果
