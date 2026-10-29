# -*- coding: utf-8 -*-
"""
03_train.py   模型训练脚本
=======================================================================
用法（在 code 目录下运行）：
    python 03_train.py --model improved --epochs 50        # 训练改进网络（最终提交用这个）
    python 03_train.py --model baseline --epochs 30        # 训练基线网络（用于修改前后对比）

流程：
    读取 dataset/train、dataset/val
    -> 训练集做数据增强（随机裁剪 + 水平翻转 + 颜色抖动 + 随机擦除）
    -> 前向/反向传播，每个 epoch 结束在验证集上评估
    -> 验证集准确率创新高就保存权重 weights/<模型名>_best.pth
    -> 训练结束后保存 history_<模型名>.json（损失/准确率历史）并画曲线

关键训练技巧（报告里会解释）：
    * CrossEntropyLoss(label_smoothing=0.1)：标签平滑，缓解过自信、抑制过拟合；
    * SGD + momentum=0.9 + weight_decay=5e-4：小数据集上泛化通常优于 Adam；
    * CosineAnnealingLR：学习率余弦退火，先大后小，后期稳定收敛；
    * 验证集只做归一化、不做增强，保证评估公平。
=======================================================================
"""

import argparse
import json
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
models_mod = import_module("02_models")     # 文件名以数字开头，用 import_module 导入
build_model, count_parameters, MODEL_DESC = (
    models_mod.build_model, models_mod.count_parameters, models_mod.MODEL_DESC)

# ------------------------- 路径配置 -------------------------
PROJECT = os.path.dirname(HERE)
DATASET_DIR = os.path.join(PROJECT, "dataset")
WEIGHT_DIR = os.path.join(PROJECT, "weights")
RESULT_DIR = os.path.join(PROJECT, "results")
# -----------------------------------------------------------

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG_SIZE = 32
MEAN = [0.4914, 0.4822, 0.4465]      # CIFAR-10 训练集统计均值
STD = [0.2470, 0.2435, 0.2616]       # CIFAR-10 训练集统计标准差


class Tee:
    """把控制台输出同时写进日志文件（报告里的“训练过程截图”就来自这份日志）。"""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def pick_device_and_amp(amp_arg):
    """选择设备；CPU 且支持 AVX512 时启用 bf16 混合精度加速（结果精度不变）。"""
    if torch.cuda.is_available():
        return torch.device("cuda"), (amp_arg != "off")
    cap = ""
    try:
        cap = torch.backends.cpu.get_cpu_capability()
    except Exception:
        pass
    use_amp = (amp_arg == "on") or (amp_arg == "auto" and cap == "AVX512")
    return torch.device("cpu"), use_amp


def get_loaders(batch_size, workers, include_test=False):
    """构建 train / val 数据加载器。

    include_test=True（本作业按老师补充的新要求采用）：
        把 test 集的 10000 张也并入训练集（同样施加数据增强），
        实际训练数据 = train 45000 + test 10000 = 55000 张；
        val 5000 张始终不参与训练，只用于每轮评估和挑选最优权重。
    include_test=False（常规做法）：
        只加载 train 更新参数，test 全程不参与训练，仅最终评估时使用。
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=4),          # 随机裁剪：平移不变性
        transforms.RandomHorizontalFlip(),                   # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),  # 随机擦除小块区域
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=eval_tf)
    test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=eval_tf)
    class_names = train_set.classes            # 类别顺序（与 ImageFolder 目录名排序一致）
    sizes = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}
    if include_test:
        # 新要求：测试集参与训练 —— 用 ConcatDataset 把 train 与 test 的索引拼在一起，
        # 不复制图片文件，类别顺序完全一致，标签可直接混用
        train_set = ConcatDataset([train_set, test_set])
    # persistent_workers=True：Windows 下子进程只创建一次，避免每个 epoch 重新启动
    # 数据加载进程带来的额外开销（否则每轮会多花几十秒）
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers, persistent_workers=workers > 0)
    return train_set, val_set, train_loader, val_loader, class_names, sizes


def run_one_epoch(model, loader, criterion, device, use_amp, optimizer=None, max_batches=None):
    """跑一遍数据：optimizer 不为 None 表示训练，否则表示评估。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def plot_curve(history, name):
    """画训练/验证的损失曲线和准确率曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", ms=3, label="训练损失 train loss")
    axes[0].plot(epochs, history["val_loss"], "s-", ms=3, label="验证损失 val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("损失曲线 Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", ms=3, label="训练准确率 train acc")
    axes[1].plot(epochs, history["val_acc"], "s-", ms=3, label="验证准确率 val acc")
    axes[1].axhline(0.85, ls="--", c="r", lw=1, label="作业要求 85%")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("准确率曲线 Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULT_DIR, f"curve_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[OK] 训练曲线已保存：{path}")


def main():
    parser = argparse.ArgumentParser(description="图像分类模型训练")
    parser.add_argument("--model", type=str, default="improved", choices=["baseline", "improved"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印数据清单（各集合张数、训练轮数），不真正训练")
    parser.add_argument("--eval-test", action=argparse.BooleanOptionalAction, default=True,
                        help="每个 epoch 都在测试集上评估并打印 test_loss / test_acc（默认开启，"
                             "与课堂示范代码的输出格式保持一致；用 --no-eval-test 关闭）")
    parser.add_argument("--tag", type=str, default="",
                        help="输出文件后缀，例如 _demo，避免覆盖正式权重与日志")
    parser.add_argument("--include-test", action="store_true",
                        help="把测试集并入训练集（本作业按老师新要求采用该方式）")
    parser.add_argument("--force", action="store_true",
                        help="允许在“完全不带参数”的情况下开始训练（防止误触 Run 覆盖已训练权重）")
    args = parser.parse_args()

    # ---------- 误触保护 ----------
    # 在 PyCharm / IDE 里直接点 Run 时往往“不带任何命令行参数”，而默认参数是
    # 50 个 epoch 且不含 --include-test，一旦误触会把正式权重和训练日志全部覆盖。
    # 因此这里要求：完全不带参数时必须加 --force 才继续训练。
    if len(sys.argv) == 1 and not args.force:
        print("=" * 84)
        print("[已拦截] 检测到你在“不带任何参数”的情况下运行本脚本。")
        print("         默认参数是 improved + 50 个 epoch + 测试集不参与训练，")
        print("         直接跑会覆盖 weights/improved_best.pth 与 results/train_log_improved.txt。")
        print("-" * 84)
        print("正确用法（复制到 PyCharm 底部 Terminal 或命令行执行）：")
        print("   python 03_train.py --model improved --epochs 24 --include-test")
        print("   python 03_train.py --model improved --epochs 24 --include-test --dry-run  # 只看数据清单")
        print("   只想试跑、不覆盖正式结果：加 --tag _demo")
        print("确实要用默认参数训练：加 --force")
        print("=" * 84)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    device, use_amp = pick_device_and_amp(args.amp)
    # 只有真正训练时才写日志文件；--dry-run 只打印到屏幕，绝不覆盖已有的训练日志
    if args.dry_run:
        tee = None
    else:
        log_name = f"train_log_{args.model}{args.tag}.txt"
        tee = Tee(os.path.join(RESULT_DIR, log_name))
    if tee is not None:
        sys.stdout = tee

    train_set, val_set, train_loader, val_loader, class_names, sizes = get_loaders(
        args.batch_size, args.workers, include_test=args.include_test)
    model = build_model(args.model, num_classes=len(class_names))
    total, trainable, head = count_parameters(model)

    data_source = ("train + test（测试集已并入训练）" if args.include_test
                   else "train（测试集不参与训练，仅最终评估时使用）")
    n_epoch_images = len(train_set) * args.epochs
    print("=" * 84)
    print(f"模型         : {MODEL_DESC[args.model]}")
    print(f"参数量       : 总 {total:,} / 可训练 {trainable:,} / 分类头 {head:,}")
    print(f"设备         : {device}   混合精度(bf16): {use_amp}   CPU 线程: {args.threads}")
    print(f"训练数据来源 : {data_source}")
    print("-" * 84)
    print(f"数据清单     : 训练集 train {sizes['train']} 张 | 验证集 val {sizes['val']} 张 | "
          f"测试集 test {sizes['test']} 张")
    print(f"               （测试集 {sizes['test']} 张{'已并入训练' if args.include_test else '不参与训练'}，"
          f"每类 {sizes['val'] // max(len(class_names), 1)} 张验证图）")
    print(f"实际参与训练 : {len(train_set)} 张（类别数 {len(class_names)}）")
    print(f"训练轮数     : {args.epochs} 个 epoch"
          + (f" —— 测试集将参与 {args.epochs} 轮训练" if args.include_test else "")
          + f"，累计喂入约 {n_epoch_images:,} 张图片（{args.epochs} × {len(train_set)}）")
    print(f"超参数       : 批大小 {args.batch_size}   初始学习率 {args.lr}   "
          f"weight_decay: {args.weight_decay}   损失: 交叉熵+标签平滑0.1")
    print(f"模型保存路径 : weights/{args.model}{args.tag}_best.pth")
    print("=" * 84)

    if args.dry_run:
        print("[dry-run] 只打印数据清单，未进行训练（不写日志文件、不覆盖任何已有结果）。"
              "去掉 --dry-run 即开始正式训练。")
        if tee is not None:
            sys.stdout = tee.stdout
            tee.close()
        return

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)      # 标签平滑
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    weight_path = os.path.join(WEIGHT_DIR, f"{args.model}{args.tag}_best.pth")

    # --eval-test：每个 epoch 额外在测试集（10000 张）上评估一次，打印 test_loss / test_acc
    test_eval_loader = None
    if args.eval_test:
        test_eval_set = datasets.ImageFolder(
            os.path.join(DATASET_DIR, "test"),
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
        test_eval_loader = DataLoader(test_eval_set, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.workers)
        print(f"每轮测试集评估 : 已开启（测试集 {len(test_eval_set)} 张，共 {args.epochs} 轮）")
        print("=" * 84)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "test_loss": [], "test_acc": [], "lr": [], "time": []}
    best_acc, best_epoch = 0.0, 0
    total_time = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, device, use_amp, optimizer)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, device, use_amp)
        if test_eval_loader is not None:
            te_loss, te_acc = run_one_epoch(model, test_eval_loader, criterion, device, use_amp)
        else:
            te_loss, te_acc = 0.0, 0.0
        scheduler.step()
        cost = time.time() - start
        total_time += cost

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["lr"].append(cur_lr)
        history["time"].append(cost)

        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save({"state_dict": model.state_dict(), "model_name": args.model,
                        "classes": class_names, "img_size": IMG_SIZE,
                        "val_acc": va_acc, "epoch": epoch, "mean": MEAN, "std": STD,
                        "include_test": bool(args.include_test),
                        "test_acc": te_acc,
                        "train_size": len(train_set)},
                       weight_path)
            flag = "  <-- 保存最优模型"

        if test_eval_loader is not None:
            # 与课堂示范代码一致的输出格式：每一轮同时给出训练集与测试集的 loss / acc
            print(f"[Epoch {epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.2%} | "
                  f"test_loss={te_loss:.4f} test_acc={te_acc:.2%} | {cost:.1f}s{flag}", flush=True)
        else:
            print(f"[Epoch {epoch:02d}/{args.epochs}] lr={cur_lr:.4f} | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2%} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.2%} | {cost:.1f}s{flag}", flush=True)

    print("=" * 84)
    print(f"训练结束！最优验证准确率 {best_acc:.2%}（第 {best_epoch} 个 epoch），"
          f"总耗时 {total_time / 60:.1f} 分钟")
    print(f"权重文件：weights/{args.model}{args.tag}_best.pth")

    history["best_val_acc"] = best_acc
    history["best_epoch"] = best_epoch
    history["total_time_min"] = total_time / 60
    history["model"] = args.model
    history["params"] = total
    history["head_params"] = head
    history["epochs"] = args.epochs
    history["train_size"] = len(train_set)
    history["val_size"] = len(val_set)
    history["data_source"] = data_source
    with open(os.path.join(RESULT_DIR, f"history_{args.model}{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_curve(history, f"{args.model}{args.tag}")
    if tee is not None:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
