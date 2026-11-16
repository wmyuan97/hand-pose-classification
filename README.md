# 手部姿态分类：传统方法（54 张）

## 数据

- 原参与者共 54 张图片：左手 27 张、右手 27 张。
- 分类：外展（ABD）、内收（ADD）、对掌（OPP）。
- 拍摄角度：正面、45°、90°。
- 遮挡条件：无遮挡、纸张遮挡、另一只手遮挡。
- 本版本不包含新增参与者数据，也不使用 AI 方法。
- 公开仓库不包含参与者原始照片；经授权的数据应按照 `labels.csv` 中的路径放置。

## 处理方法

1. 使用 YCrCb 和 HSV 颜色空间分割皮肤区域。
2. 通过形态学操作去噪，并提取主要手部轮廓。
3. 将右手镜像到统一方向，并归一化手部区域。
4. 提取轮廓、凸包、Hu 矩、凸缺陷、掌心半径、骨架、宽度分布和 HOG 特征。
5. 分别测试 SVM、Random Forest、梯度提升和规则分类方法。
6. 使用分层六折交叉验证，并测试不同角度和遮挡条件。

## 目录和脚本说明

```text
project_traditional_54/
├── 左手/、右手/                         原参与者的 54 张图片
├── labels.csv                           图片标签和路径
├── requirements.txt                     Python 依赖
└── traditional_method/
    ├── segment_hands.py                 皮肤分割、去噪并生成手部掩膜
    ├── extract_features.py              提取轮廓、形状和 Hu 矩特征
    ├── train_svm.py                     训练并评估基础 SVM
    ├── enhanced_traditional.py          测试增强特征、Random Forest 和梯度提升
    ├── evaluate_rule_based.py           测试人工阈值规则分类方法
    ├── analyze_features.py              生成特征统计、箱线图和 PCA 分析
    ├── analyze_svm_errors.py             分析 SVM 的错误分类图片
    ├── analyze_hand_geometry.py          可视化掌心、凸缺陷和拇指几何关系
    └── results/                          保存掩膜、特征、预测表和结果图
```

主要运行顺序：

```bash
python traditional_method/segment_hands.py
python traditional_method/extract_features.py
python traditional_method/train_svm.py
python traditional_method/enhanced_traditional.py
python traditional_method/evaluate_rule_based.py
```

## 实验结果

| 方法 | 六折验证 | 纸张遮挡 | 手部遮挡 | 正面 | 45° | 90° |
|---|---:|---:|---:|---:|---:|---:|
| 基础 SVM | 42.59% | 66.67% | 27.78% | 50.00% | 61.11% | 33.33% |
| 增强几何 SVM | 44.44% | 55.56% | 33.33% | 50.00% | 44.44% | 22.22% |
| Random Forest | 48.15% | 66.67% | **50.00%** | **55.56%** | **61.11%** | **44.44%** |
| 梯度提升 | **51.85%** | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% |
| HOG SVM | 38.89% | 55.56% | 22.22% | 44.44% | 44.44% | 27.78% |

规则分类方法的总体准确率为 **37.04%**。

## 结论

- 梯度提升的六折验证结果最高，为 **51.85%**。
- Random Forest 在不同角度和遮挡测试中更稳定，90°准确率由基础 SVM 的 33.33% 提升至 **44.44%**。
- 90°侧视图和手部遮挡仍然是传统方法的主要困难。

结果文件保存在 `traditional_method/results/`。
