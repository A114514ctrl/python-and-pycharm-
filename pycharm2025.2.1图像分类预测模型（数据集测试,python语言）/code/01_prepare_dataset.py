# -*- coding: utf-8 -*-
"""
01_prepare_dataset.py   数据集制作与划分
=======================================================================
选题：选题2 图像分类模型
数据来源：公开数据集 CIFAR-10（Canadian Institute For Advanced Research）
          官网 https://www.cs.toronto.edu/~kriz/cifar.html
          CIFAR-10 共 60000 张 32x32 彩色真实照片、10 个类别，
          官方划分为 50000 张训练图 + 10000 张测试图。

本脚本做四件事：
  1) 获取 CIFAR-10 原始数据：
       优先使用 data_raw/ 下已有的数据文件；
       否则自动下载（默认使用 HuggingFace 镜像，速度较快；--source official
       则使用 torchvision 官方源，与课堂代码完全一致）；
  2) 把原始数据导出成 ImageFolder 可直接读取的图片目录：
        dataset/train/<类别>/*.png   45000 张（每类 4500 张）
        dataset/val/<类别>/*.png      5000 张（每类  500 张）
        dataset/test/<类别>/*.png    10000 张（每类 1000 张）
     划分方式：在官方 50000 张训练图中，按类别做 9:1 分层抽样得到 train / val；
              官方 10000 张测试图全部作为 test。三者完全独立、互不重叠；
  3) 统计数据集规模、类别分布，写入 results/dataset_stats.json；
  4) 画出三张图（作业可截图）：
        results/dataset_distribution.png  各类别数量柱状图
        results/dataset_preview.png       每类 8 张样本预览
        results/augmentation_preview.png  数据增强效果预览

运行（Python 3.10）：
    python 01_prepare_dataset.py
    python 01_prepare_dataset.py --source official      # 用官方源下载
=======================================================================
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request

import numpy as np
from PIL import Image

# ------------------------- 可修改的参数 -------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # code 目录
PROJECT = os.path.dirname(HERE)                            # 项目根目录
RAW_DIR = os.path.join(PROJECT, "data_raw")                # 原始数据缓存
DATASET_DIR = os.path.join(PROJECT, "dataset")             # 导出的图片数据集
RESULT_DIR = os.path.join(PROJECT, "results")              # 图表输出目录
VAL_PER_CLASS = 500                                        # 每类划出 500 张做验证集
SEED = 42
# CIFAR-10 类别顺序（官方固定顺序，标签 0~9 与之一一对应）
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CLASS_CN = {"airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫", "deer": "鹿",
            "dog": "狗", "frog": "青蛙", "horse": "马", "ship": "轮船", "truck": "卡车"}
# HuggingFace 镜像上的 CIFAR-10 数据文件（与官方内容完全一致）
MIRROR_FILES = {
    "train": "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
             "plain_text/train-00000-of-00001.parquet",
    "test": "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
            "plain_text/test-00000-of-00001.parquet",
}
# ---------------------------------------------------------------


def setup_matplotlib():
    """matplotlib 中文字体设置（Windows 用微软雅黑）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


class Tee:
    """把控制台输出同时写入日志文件 results/console_prepare.txt
    （报告里的“数据集制作过程截图”就来自这份日志，用 utf-8 保存避免中文乱码）。"""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
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


# ==================== 一、获取原始数据 ====================
def download_mirror(split):
    """从镜像下载 parquet 数据文件（带进度打印）。"""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{split}-00000-of-00001.parquet")
    if os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
        print(f"      已存在 {os.path.basename(path)}（{os.path.getsize(path) / 1e6:.1f} MB），跳过下载")
        return path
    url = MIRROR_FILES[split]
    print(f"      正在下载 {split} 数据：{url}")

    def hook(blocks, block_size, total_size):
        if total_size > 0:
            done = min(blocks * block_size / total_size, 1.0)
            sys.stdout.write(f"\r      进度 {done * 100:5.1f}%  "
                             f"({blocks * block_size / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, path, reporthook=hook)
    print()
    return path


def load_from_parquet(source="auto"):
    """返回 {split: [(png_bytes, label), ...]}。"""
    import pandas as pd
    data = {}
    for split in ("train", "test"):
        if source == "official":
            break
        path = download_mirror(split)
        df = pd.read_parquet(path)                     # 两列：img(含 png 字节) / label(0~9)
        images = df["img"].tolist()
        labels = df["label"].tolist()
        data[split] = [(d["bytes"], int(lab)) for d, lab in zip(images, labels)]
        print(f"      {split} 读取完成：{len(data[split])} 张")
    return data


def load_from_official():
    """用 torchvision 官方接口下载并读取 CIFAR-10（与课堂代码一致的方式）。"""
    from torchvision import datasets
    os.makedirs(RAW_DIR, exist_ok=True)
    print("      正在使用 torchvision 官方源下载 CIFAR-10（约 163MB，可能较慢）...")
    train_set = datasets.CIFAR10(root=RAW_DIR, train=True, download=True)
    test_set = datasets.CIFAR10(root=RAW_DIR, train=False, download=True)

    data = {}
    for split, ds in (("train", train_set), ("test", test_set)):
        rows = []
        for i in range(len(ds)):
            img, label = ds[i]                       # img 是 PIL.Image
            rows.append((img.tobytes(), int(label), img.size))
        data[split] = rows
        print(f"      {split} 读取完成：{len(rows)} 张")
    return data


# ==================== 二、导出图片并划分数据集 ====================
def save_png(png_bytes, out_path):
    """把 PNG 字节流直接写成文件（不重新编码，保证与原始数据完全一致）。"""
    with open(out_path, "wb") as f:
        f.write(png_bytes)


def export_split(items, split_name, indices):
    """把指定下标的样本存成 dataset/<split_name>/<类别>/xxx.png，返回每类数量。"""
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(DATASET_DIR, split_name, name), exist_ok=True)
    counters = {name: 0 for name in CLASS_NAMES}
    for n, idx in enumerate(indices, 1):
        png_bytes, label = items[idx]
        name = CLASS_NAMES[label]
        out = os.path.join(DATASET_DIR, split_name, name, f"{name}_{counters[name]:05d}.png")
        save_png(png_bytes, out)
        counters[name] += 1
        if n % 5000 == 0:
            print(f"      {split_name}: 已导出 {n}/{len(indices)} 张 ...")
    return counters


def build_dataset(data):
    """按每类 9:1 分层抽样划分 train / val，并把官方 test 全量导出。"""
    print("[2/4] 导出图片并划分 train / val / test ...")
    if os.path.isdir(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)                    # 清理旧结果，保证可重复
    os.makedirs(DATASET_DIR, exist_ok=True)

    train_items = data["train"]
    labels = np.array([lab for _, lab in train_items])
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for cls in range(len(CLASS_NAMES)):
        cls_idx = np.where(labels == cls)[0]
        cls_idx = rng.permutation(cls_idx)            # 打乱后再切分，避免顺序偏差
        val_idx.extend(cls_idx[:VAL_PER_CLASS].tolist())
        train_idx.extend(cls_idx[VAL_PER_CLASS:].tolist())
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    train_cnt = export_split(train_items, "train", train_idx)
    val_cnt = export_split(train_items, "val", val_idx)
    test_cnt = export_split(data["test"], "test", range(len(data["test"])))

    stats = {
        "data_source": "CIFAR-10 公开数据集（https://www.cs.toronto.edu/~kriz/cifar.html）",
        "image_size": [32, 32],
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_names_cn": [CLASS_CN[c] for c in CLASS_NAMES],
        "split_rule": "官方 50000 张训练图按每类 9:1 分层抽样得到 train/val；"
                      "官方 10000 张测试图全部作为 test；三者互不重叠",
        "train": {"total": sum(train_cnt.values()), "per_class": train_cnt},
        "val": {"total": sum(val_cnt.values()), "per_class": val_cnt},
        "test": {"total": sum(test_cnt.values()), "per_class": test_cnt},
    }
    stats["all_total"] = stats["train"]["total"] + stats["val"]["total"] + stats["test"]["total"]
    return stats


# ==================== 三、数据集相关图表 ====================
def plot_distribution(plt, stats):
    """各类别数量柱状图。"""
    names = stats["class_names"]
    labels = [f"{n}\n{cn}" for n, cn in zip(names, stats["class_names_cn"])]
    tr = [stats["train"]["per_class"][n] for n in names]
    va = [stats["val"]["per_class"][n] for n in names]
    te = [stats["test"]["per_class"][n] for n in names]

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    b1 = ax.bar(x - w, tr, w, label=f"训练集 train（{stats['train']['total']} 张）")
    b2 = ax.bar(x, va, w, label=f"验证集 val（{stats['val']['total']} 张）")
    b3 = ax.bar(x + w, te, w, label=f"测试集 test（{stats['test']['total']} 张）")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fontsize=6.5, padding=1)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("图片数量 / 张")
    ax.set_title(f"CIFAR-10 数据集类别分布（总计 {stats['all_total']} 张，{stats['num_classes']} 类）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, "dataset_distribution.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"      已保存类别分布图：{path}")


def plot_preview(plt, stats, cols=8):
    """每类抽 8 张样本预览。"""
    names = stats["class_names"]
    fig, axes = plt.subplots(len(names), cols, figsize=(cols * 1.25, len(names) * 1.5))
    for r, name in enumerate(names):
        folder = os.path.join(DATASET_DIR, "train", name)
        files = sorted(os.listdir(folder))[:cols]
        for c, fn in enumerate(files):
            ax = axes[r][c]
            ax.imshow(Image.open(os.path.join(folder, fn)))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
            if c == 0:
                ax.set_ylabel(f"{name}\n{CLASS_CN.get(name, '')}", fontsize=10)
    fig.suptitle("训练集样本预览（真实照片，每类随机 8 张）", fontsize=13)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, "dataset_preview.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"      已保存样本预览图：{path}")


def plot_augmentation(plt):
    """数据增强效果预览：同一张图的 8 个随机增强变体。"""
    from torchvision import transforms
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    folder = os.path.join(DATASET_DIR, "train", "cat")
    src = Image.open(os.path.join(folder, sorted(os.listdir(folder))[0]))
    fig, axes = plt.subplots(1, 9, figsize=(13.5, 2.0))
    axes[0].imshow(src)
    axes[0].set_title("原图", fontsize=10)
    axes[0].set_xticks([]), axes[0].set_yticks([])
    for i in range(1, 9):
        axes[i].imshow(aug(src))
        axes[i].set_title(f"增强{i}", fontsize=9)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    fig.suptitle("训练时的数据增强（随机裁剪 + 水平翻转 + 颜色抖动），用于抑制过拟合", fontsize=12)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, "augmentation_preview.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"      已保存数据增强预览图：{path}")


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 数据集导出与划分")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "mirror", "official"],
                        help="数据获取方式：auto/mirror=镜像，official=torchvision 官方源")
    args = parser.parse_args()

    np.random.seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt = setup_matplotlib()

    tee = Tee(os.path.join(RESULT_DIR, "console_prepare.txt"))
    sys.stdout = tee
    try:
        print("[1/4] 获取 CIFAR-10 原始数据 ...")
        if args.source == "official":
            data = load_from_official()
        else:
            try:
                data = load_from_parquet(args.source)
            except Exception as exc:               # 镜像不可用时自动回退到官方源
                print(f"      镜像获取失败（{exc}），改用 torchvision 官方源 ...")
                data = load_from_official()

        stats = build_dataset(data)

        print("[3/4] 统计数据集信息 ...")
        print(f"      总图片数 {stats['all_total']} 张 / {stats['num_classes']} 类")
        print(f"      train {stats['train']['total']} 张，val {stats['val']['total']} 张，"
              f"test {stats['test']['total']} 张")
        print(f"      每类数量：train {stats['train']['per_class']['cat']} 张、"
              f"val {stats['val']['per_class']['cat']} 张、test {stats['test']['per_class']['cat']} 张")
        with open(os.path.join(RESULT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("[4/4] 绘制数据集相关图表 ...")
        plot_distribution(plt, stats)
        plot_preview(plt, stats)
        plot_augmentation(plt)
        print("数据集制作完成！目录：", DATASET_DIR)
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
