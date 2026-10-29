# -*- coding: utf-8 -*-
"""
02_models.py   网络模型定义（修改前 / 修改后）
=======================================================================
本文件放两个模型，训练脚本和测试脚本都从这里 import，避免重复写代码：

1) BaselineCNN —— “课堂原始网络”（修改前）
   就是上课讲的经典小卷积网络写法：3 个“卷积+BN+ReLU+池化”的堆叠，
   通道数 16 / 32 / 64，最后接一个大全连接层(64*4*4 -> 128)做分类。
   它结构简单、参数少，在小数据集（比如作业2的几何图形）上够用，
   但放到 CIFAR-10 这种真实的 10 类自然图像上，表达能力不足，
   测试准确率通常在 75% 左右，达不到 85% 的要求。

2) ImprovedCNN —— “改进网络”（修改后）
   针对 BaselineCNN 的问题做了 6 处修改（报告里逐条说明）：
   ① 加深加宽：卷积层 3 层 -> 8 层，通道 16/32/64 -> 32/64/128/256；
   ② 每个卷积块内部堆叠 2 个 3x3 卷积（扩大感受野，特征更抽象）；
   ③ 末段加入 1x1 卷积做通道变换，参数少、非线性更强；
   ④ 用全局平均池化(GAP) + 1 个 Dropout 全连接 替换原来的大全连接层，
      分类头参数从 65k 降到 2.5k，明显更“轻量化”，并且抗过拟合；
   ⑤ 统一使用 3x3 小卷积核（相比 5x5 参数更少、层数可以更深）；
   ⑥ 训练侧配合：标签平滑 + 余弦退火学习率 + 数据增强（在 03_train.py 中）。

模型输入统一为 3x32x32（CIFAR-10 原图尺寸），输出 10 类 logits。
=======================================================================
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """修改前的基线网络（课堂原始网络写法）：3 层卷积 + 大全连接层。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 3x32x32 -> 16x16x16
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16x16 -> 32x8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x8x8 -> 64x4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """修改后的网络：更深的 3x3 卷积堆叠 + 1x1 卷积 + 全局平均池化分类头。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        def conv_bn(cin, cout, kernel=3):
            """卷积 + BN + ReLU 的标准小单元（bias=False，因为 BN 自带偏置）。"""
            pad = kernel // 2
            return [nn.Conv2d(cin, cout, kernel, padding=pad, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            # 32x32x3  -> 32x32x32   （第 1 个 3x3 卷积，不立刻降采样）
            *conv_bn(3, 32),
            # 32x32x32 -> 16x16x32   （块内两个 3x3 卷积 + 池化）
            *conv_bn(32, 32), *conv_bn(32, 32), nn.MaxPool2d(2),
            # 16x16x32 -> 8x8x64
            *conv_bn(32, 64), *conv_bn(64, 64), nn.MaxPool2d(2),
            # 8x8x64   -> 4x4x128
            *conv_bn(64, 128), *conv_bn(128, 128), nn.MaxPool2d(2),
            # 4x4x128  -> 4x4x256   （1x1 卷积做通道升维，等价于轻量全连接）
            *conv_bn(128, 256, kernel=1),
            # 4x4x256  -> 256x1x1   （全局平均池化，替代原来的 Flatten + 大 FC）
            nn.AdaptiveAvgPool2d(1),
        )
        # 分类头：只有 256*10+10 个参数，非常轻量
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# 模型名 -> 类，训练/测试脚本用字符串指定，方便命令行切换
MODELS = {"baseline": BaselineCNN, "improved": ImprovedCNN}

MODEL_DESC = {
    "baseline": "BaselineCNN（修改前：课堂原始网络，3 层卷积 16/32/64 + 全连接 128）",
    "improved": "ImprovedCNN（修改后：8 层卷积 32/64/128/256 + 1x1 卷积 + 全局平均池化）",
}


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """按名字创建模型。"""
    if name not in MODELS:
        raise ValueError(f"未知模型 {name}，可选：{list(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module):
    """统计参数量：总参数 / 可训练参数 / 分类头参数。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.classifier.parameters()) if hasattr(model, "classifier") else 0
    return total, trainable, head


if __name__ == "__main__":
    # 直接运行本文件可以快速检查两个网络的结构和参数量
    for name in MODELS:
        model = build_model(name)
        total, trainable, head = count_parameters(model)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print("=" * 70)
        print(f"{name:<9} 输出形状 {tuple(y.shape)} | 总参数 {total:,} | "
              f"可训练 {trainable:,} | 分类头 {head:,}")
        print(model)
