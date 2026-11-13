# CNN 大作业：Fashion-MNIST 轻量 CNN 消融实验与退化场景鲁棒性评估

本项目在 Fashion-MNIST 上实现一个 LeNet 风格的精简 CNN，围绕 CNN 核心组件做 5 组受控消融实验，并在 10 种退化场景下评估各模型的鲁棒性。全部实验在 CPU 上可复现。

## 1. 实验内容

| 编号 | 模型 | 结构改动 |
| :---: | :--- | :--- |
| 1 | Baseline | 标准 3×3 卷积 + BatchNorm + ReLU |
| 2 | Depthwise Separable | 标准卷积替换为 DW 3×3 + PW 1×1 |
| 3 | SE Attention | 每个 ConvBlock 后加 SE 通道注意力 |
| 4 | GELU Activation | ReLU 替换为 GELU |
| 5 | GroupNorm | BatchNorm 替换为 GroupNorm(8 组) |

基础架构 `SmallCNN`：3 个 ConvBlock，通道 32→64→128，每个 Block 后接 MaxPool，最后全局平均池化 + 全连接分类头。所有变体共享 `models.py` 中的同一套代码，通过参数开关切换。

## 2. 目录结构

```text
CNN/
├── requirements.txt        # 依赖环境
├── README.md               # 本文件
├── models.py               # 模型结构定义（SmallCNN, SEBlock, DepthwiseSeparableConv）
├── utils.py                # 工具函数（随机种子、参数量、MACs 统计）
├── train10.py              # 早期 10 epoch 训练脚本（生成 results10.csv，用于对比）
├── train20.py              # 最终 20 epoch 训练脚本（生成 results20.csv 和 checkpoints/）
├── scenario_eval.py        # 退化场景鲁棒性评估脚本（生成 scenario_results.csv）
├── plot.py                 # 绘图脚本（读取 csv 生成 figures/ 下的对比图）
├── results10.csv           # 10 epoch 主实验结果
├── results20.csv           # 20 epoch 最终主实验结果（报告核心数据）
├── scenario_results.csv    # 场景鲁棒性评估结果（报告场景应用数据）
├── checkpoints/            # 最佳权重目录（不提交至 Git，运行时生成）
├── figures/                # 对比图输出目录（不提交至 Git，运行时生成）
└── data/                   # Fashion-MNIST 数据集目录（不提交至 Git，自动下载）
```

## 3. 环境与安装

建议使用虚拟环境：

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证环境：

```bash
python -c "import torch, torchvision, thop, pandas, matplotlib; print('ok')"
```

## 4. 运行步骤

首次运行会自动下载 Fashion-MNIST 到 `./data`（约 30 MB）。

**步骤 1：训练 20 epoch 最终模型（核心）**

```bash
python train20.py
```

输出：`results20.csv`、`checkpoints/*.pt`（5 个 best 权重）

说明：固定使用 10% 训练集作验证集选 best，test set 仅在最终评估使用一次。CPU 上 5 组 × 20 epoch 约 75～85 分钟。

**步骤 2：退化场景鲁棒性评估**

```bash
python scenario_eval.py
```

输出：`scenario_results.csv`

说明：加载 `checkpoints/` 下的权重，在 10 种退化场景（噪声、模糊、遮挡、亮度）下评估。CPU 上约 5 分钟。

**步骤 3：生成报告图表**

```bash
python plot.py
```

输出：`figures/` 下生成 4 张对比图

说明：自动读取 `results10.csv`、`results20.csv` 和 `scenario_results.csv`。中文显示依赖系统字体（默认设置了 SimHei）。

> **注**：如需复现早期 10 epoch 结果，可运行 `python train10.py`，生成 `results10.csv`。

## 5. 核心结果摘要

主实验结果（来自 `results20.csv`）：

- **Baseline**：参数量 0.0942M，MACs 7.63M，Test Acc 90.22%。
- **GELU**：参数量 0.0942M，MACs 7.63M，Test Acc 90.35%。
- **Depthwise Separable**：参数量 0.0129M（降低 86.3%），MACs 1.10M（降低 85.6%），Test Acc 84.05%。

场景鲁棒性分析（来自 `scenario_results.csv`）：

- **GN（GroupNorm）**：在全局亮度/尺度变化下体现出极强的尺度不变性，准确率几乎不掉点。
- **DW（深度可分离卷积）**：在噪声、模糊、遮挡场景下掉点幅度约为基线的一半，体现出小容量模型抗局部扰动能力更强。

## 6. 实验设置明细

- **数据集**：Fashion-MNIST（1×28×28，10 类）
- **数据增强**：RandomHorizontalFlip + RandomCrop(28, padding=2)
- **优化器**：Adam (lr=1e-3, weight_decay=5e-4)
- **学习率调度**：CosineAnnealingLR
- **Batch Size**：128
- **随机种子**：固定 seed=42
- **选模型策略**：按验证集准确率选最佳 epoch 权重，再在 test 集评估一次