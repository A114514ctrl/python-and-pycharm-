# CIFAR-10 图像分类（改进 CNN）

用 PyTorch 从零实现的一个图像分类项目：把课堂上讲的 3 层小卷积网络改进为 8 层卷积网络，
在 CIFAR-10 数据集上完成训练、测试与可视化，并自动生成本地 Word 实验报告。

## 结果

| 指标 | 基线网络 BaselineCNN | 改进网络 ImprovedCNN |
| --- | --- | --- |
| 卷积层数 | 3 | 8 |
| 通道数 | 16 / 32 / 64 | 32 / 64 / 128 / 256 |
| 总参数量 | 156,298 | 332,586 |
| 分类头参数 | 132,490 | **2,570（-98%）** |
| 训练轮数 | 20 epochs | 24 epochs |
| 测试集准确率 | 79.57% | **89.72%** |
| 宏平均 F1 | 0.7943 | **0.8972** |

## 网络改进点

1. 卷积层 3 → 8 层（加深）
2. 通道 16/32/64 → 32/64/128/256（加宽）
3. 每个分辨率阶段堆叠 2 个 3×3 卷积（扩大感受野且参数更少）
4. 末端新增 1×1 卷积（通道升维，参数极少）
5. 全局平均池化替代 Flatten + 大全连接（分类头参数 -98%，轻量化且抗过拟合）
6. 训练策略：标签平滑 0.1、余弦退火、随机裁剪/翻转/颜色抖动/随机擦除

## 目录结构

```
CIFAR10-Image-Classification/
├─ code/
│   ├─ 01_prepare_dataset.py     数据集下载、导出为 ImageFolder 格式、按类 9:1 分层划分
│   ├─ 02_models.py              网络定义：BaselineCNN（基线）/ ImprovedCNN（改进）
│   ├─ 03_train.py               训练脚本（增强、标签平滑、余弦退火、bf16、保存最优权重）
│   ├─ 04_test_and_visualize.py   测试评估、混淆矩阵、三行预测可视化、修改前后对比
│   ├─ 05_predict.py             交互式预测（拖图 / 回车选图 / 命令行传路径）
│   └─ 06_make_report.py          读取真实结果自动生成本地 Word 实验报告
├─ scripts/                      一键 .bat 脚本（Windows）
├─ weights/                      训练好的权重（improved_best.pth / baseline_best.pth）
├─ results/                      图表、日志、指标 json（运行后生成）
├─ report/                       自动生成的 Word 报告
├─ requirements.txt
└─ 操作说明.md                    详细使用说明
```

## 环境

- Python **3.10**
- PyTorch 2.x + torchvision（CPU 可跑，有 CUDA 会自动用 GPU）
- NumPy、Matplotlib、Pillow、pandas、pyarrow（读原始数据）、python-docx（生成报告）

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1) 准备数据集（自动下载 CIFAR-10，约 140MB；导出为 dataset/train|val|test）
python code/01_prepare_dataset.py

# 2) 训练（改进网络 24 轮；基线 20 轮用于对比）
python code/03_train.py --model improved --epochs 24 --no-eval-test
python code/03_train.py --model baseline --epochs 20 --no-eval-test

# 3) 测试集评估 + 可视化 + 前后对比
python code/04_test_and_visualize.py --model improved
python code/04_test_and_visualize.py --model baseline
python code/04_test_and_visualize.py --model improved --compare

# 4) 交互式预测（把任意图片拖进终端回车）
python code/05_predict.py

# 5) 生成 Word 实验报告（可选）
python code/06_make_report.py
```

Windows 用户也可以直接双击 `scripts/` 下的 bat：`run_inventory.bat`、`run_eval.bat`、
`run_predict.bat`、`make_report.bat`、`train_demo_2ep.bat`、`train_full_24ep.bat`、`watch_log.bat`。

## 数据划分

| 划分 | 来源 | 数量 | 用途 |
| --- | --- | --- | --- |
| train | 官方 50000 张训练图按类 9:1 抽样 | 45,000 | 更新网络参数 |
| val | 同上（每类 500 张） | 5,000 | 每轮评估、挑选最优权重 |
| test | 官方 10000 张测试图 | 10,000 | 仅在最终评估时使用一次 |

三份数据互不重叠（随机种子固定为 42，可复现）；训练阶段不接触测试集。

## 说明

- 数据集较大，`.gitignore` 默认忽略 `data/`、`dataset/`、`data_raw/` 与中间产物；
  运行 `01_prepare_dataset.py` 会自动重新生成。
- 训练在 CPU 上也能跑：改进网络 24 轮约 25 分钟（8 核，bf16 混合精度）。
