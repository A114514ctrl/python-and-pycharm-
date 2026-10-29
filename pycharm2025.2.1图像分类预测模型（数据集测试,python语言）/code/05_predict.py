# -*- coding: utf-8 -*-
"""
05_predict.py   交互式图像分类预测（现场验收用）
=======================================================================
用法（在 code 目录下运行）：

  1) 交互模式（推荐现场演示）：
         python 05_predict.py
     程序会不断提示“请输入图片路径”，把任意一张图片拖进窗口回车即可，
     立即输出预测类别、置信度和 Top-3 概率，并把可视化结果图保存到
     results/predict_single.png。输入 q 退出。

  2) 直接预测单张图片：
         python 05_predict.py --image D:\\test\\cat.jpg

  3) 预测整个文件夹（一次给出多张图的三行结果图）：
         python 05_predict.py --folder D:\\test\\images

  4) 从测试集里随机抽 8 张做三行可视化（图片 / 预测 / 真实）：
         python 05_predict.py --demo

三行可视化说明（作业要求）：
     第一行 = 测试图片；第二行 = 预测类别 + 预测置信度；第三行 = 真实类别。
=======================================================================
"""

import argparse
import os
import random
import sys
from importlib import import_module

import matplotlib
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
models_mod = import_module("02_models")
build_model = models_mod.build_model

PROJECT = os.path.dirname(HERE)
DATASET_DIR = os.path.join(PROJECT, "dataset")
WEIGHT_DIR = os.path.join(PROJECT, "weights")
RESULT_DIR = os.path.join(PROJECT, "results")
CLASS_CN = {"airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫", "deer": "鹿",
            "dog": "狗", "frog": "青蛙", "horse": "马", "ship": "轮船", "truck": "卡车"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name="improved"):
    """加载权重和预处理参数。"""
    from torchvision import transforms
    path = os.path.join(WEIGHT_DIR, f"{model_name}_best.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到权重文件 {path}，请先运行 03_train.py 训练模型")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    class_names = ckpt["classes"]
    model = build_model(ckpt.get("model_name", model_name), num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    size = ckpt.get("img_size", 32)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(ckpt.get("mean", [0.5] * 3), ckpt.get("std", [0.5] * 3)),
    ])
    return model, class_names, tf, ckpt


def predict_image(model, class_names, tf, image_path):
    """预测单张图片，返回 (预测类别, 置信度, 全部类别概率列表)。"""
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x).float(), dim=1)[0].cpu()
    conf, idx = probs.max(0)
    return class_names[idx.item()], float(conf), probs.tolist()


def print_result(image_path, pred, conf, probs, class_names):
    """在终端打印预测结果。"""

    order = sorted(range(len(class_names)), key=lambda i: probs[i], reverse=True)
    print("-" * 66)
    print(f"图片路径 : {image_path}")
    print(f"预测类别 : {pred}（{CLASS_CN.get(pred, pred)}）")
    print(f"预测置信度: {conf:.2%}")
    print("Top-3 概率:")
    for i in order[:3]:
        print(f"    {class_names[i]:<12}{CLASS_CN.get(class_names[i], ''):<6}{probs[i]:.4f}")
    print("-" * 66)


def plot_single(image_path, pred, conf, probs, class_names, save_path=None):
    """单张图片预测可视化：左边图片，右边其余类别概率条形图。"""
    order = sorted(range(len(class_names)), key=lambda i: probs[i], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].imshow(Image.open(image_path).convert("RGB"))
    axes[0].set_title(f"输入图片\n预测: {pred}（{CLASS_CN.get(pred, pred)}）\n置信度 {conf:.2%}",
                      fontsize=11)
    axes[0].set_xticks([]), axes[0].set_yticks([])

    names = [f"{class_names[i]}" for i in order]
    vals = [probs[i] for i in order]
    colors = ["#2e75b6" if i == 0 else "#b9cbe4" for i in range(len(names))]
    bars = axes[1].barh(names[::-1], vals[::-1], color=colors[::-1])
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("预测概率")
    axes[1].set_title("10 个类别的预测概率（Softmax 输出）")
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_path = save_path or os.path.join(RESULT_DIR, "predict_single.png")
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[OK] 可视化结果已保存：{save_path}")


def plot_three_row(items, class_names, title, save_path):
    """三行可视化：第一行图片、第二行预测类别+置信度、第三行真实类别。"""
    cols = len(items)
    fig, axes = plt.subplots(3, cols, figsize=(cols * 1.62, 4.6),
                             gridspec_kw={"height_ratios": [2.4, 1.0, 1.0]})
    if cols == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]
    for c, (path, true_name, pred_name, conf) in enumerate(items):
        axes[0][c].imshow(Image.open(path).convert("RGB"))
        axes[0][c].set_xticks([]), axes[0][c].set_yticks([])
        ok = (true_name is None) or (true_name == pred_name)
        for s in axes[0][c].spines.values():
            s.set_color("green" if ok else "red")
            s.set_linewidth(2)

        axes[1][c].axis("off")
        axes[1][c].text(0.5, 0.5, f"预测: {pred_name}\n置信度 {conf:.2%}", ha="center", va="center",
                        fontsize=9.5, color="green" if ok else "red", weight="bold",
                        transform=axes[1][c].transAxes)
        axes[2][c].axis("off")
        axes[2][c].text(0.5, 0.5, "真实: " + (f"{true_name}\n({CLASS_CN.get(true_name, '')})"
                                              if true_name else "未知（现场输入图片）"),
                        ha="center", va="center", fontsize=9.5, transform=axes[2][c].transAxes)
    axes[0][0].set_ylabel("测试图片", fontsize=10)
    axes[1][0].set_ylabel("预测类别\n+置信度", fontsize=10)
    axes[2][0].set_ylabel("真实类别", fontsize=10)
    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[OK] 三行预测结果图已保存：{save_path}")


def command_demo(model, class_names, tf):
    """从测试集随机抽 8 张，画三行可视化图（用于报告）。"""
    from torchvision import datasets
    test_dir = os.path.join(DATASET_DIR, "test")
    if not os.path.isdir(test_dir):
        # 按作业“统一要求”第 4 条，提交包里不含数据集，这里给出可操作的提示而不是抛异常
        print("[!] 未找到 dataset/test 目录。")
        print("    提交包按作业要求已删除原始数据集，如需此演示请先运行：")
        print("      python code/01_prepare_dataset.py")
        print("    现场验收建议直接用单张图片预测：把图片拖进窗口回车即可。")
        return
    ds = datasets.ImageFolder(test_dir)
    picks = random.Random(0).sample(ds.samples, 8)
    items = []
    for path, true_idx in picks:
        pred, conf, _ = predict_image(model, class_names, tf, path)
        items.append((path, class_names[true_idx], pred, conf))
    plot_three_row(items, class_names, "测试集随机抽样：第一行图片 / 第二行预测类别与置信度 / 第三行真实类别",
                   os.path.join(RESULT_DIR, "prediction_grid_demo.png"))


def command_folder(model, class_names, tf, folder):
    """预测一个文件夹里的所有图片（最多 8 张），并画三行图。"""
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXTS)][:8]
    if not files:
        print(f"[!] 文件夹里没有图片：{folder}")
        return
    items = []
    for fn in files:
        path = os.path.join(folder, fn)
        pred, conf, probs = predict_image(model, class_names, tf, path)
        print_result(path, pred, conf, probs, class_names)
        items.append((path, None, pred, conf))
    plot_three_row(items, class_names, f"文件夹批量预测（{folder}）",
                   os.path.join(RESULT_DIR, "predict_folder.png"))


def clean_path(text):
    """去掉拖拽文件时带入的引号和空白。"""
    return text.strip().strip('"').strip("'").strip()


def ask_file_dialog():
    """弹出选图窗口，返回选中的图片路径（拖拽无效时作为兜底方案）。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="请选择要预测的图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("所有文件", "*.*")])
        root.destroy()
        return path or ""
    except Exception as exc:
        print(f"[!] 打不开文件选择窗口（{exc}），请手动粘贴图片路径")
        return ""


def interactive(model, class_names, tf):
    """交互模式：不断读入图片路径并预测。"""
    print("=" * 66)
    print("交互式图像分类预测")
    print("  方式1：直接把图片拖进【命令行窗口】后回车（PyCharm 请用底部 Terminal 标签，Run 窗口不接收拖拽）")
    print("  方式2：什么都不输入、直接按回车 —— 弹出选图窗口，点选一张图片")
    print("  方式3：把图片路径粘贴进来回车；输入 demo 用测试集演示；输入 q 退出")
    print(f"设备: {DEVICE}    类别数: {len(class_names)}")
    print("=" * 66)
    while True:
        try:
            raw = input("\n请输入图片路径（或输入 demo 用测试集图片演示，q 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        cmd = clean_path(raw)
        if cmd.lower() in ("q", "quit", "exit"):
            print("已退出。")
            break
        if cmd.lower() == "demo":
            command_demo(model, class_names, tf)
            continue
        if not cmd:                     # 直接回车 -> 弹出选图窗口（拖拽无效时的兜底）
            cmd = clean_path(ask_file_dialog())
            if not cmd:
                continue
        if os.path.isdir(cmd):
            command_folder(model, class_names, tf, cmd)
            continue
        if not os.path.isfile(cmd):
            print(f"[!] 找不到这个文件：{cmd}")
            continue
        pred, conf, probs = predict_image(model, class_names, tf, cmd)
        print_result(cmd, pred, conf, probs, class_names)
        plot_single(cmd, pred, conf, probs, class_names)


def main():
    parser = argparse.ArgumentParser(description="图像分类交互式预测")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--folder", type=str, default=None, help="图片文件夹路径")
    parser.add_argument("--demo", action="store_true", help="用测试集随机 8 张做三行可视化")
    parser.add_argument("--model", type=str, default="improved", choices=["baseline", "improved"])
    args = parser.parse_args()

    # 把本次运行的控制台输出同时保存到 results/console_predict.txt（报告截图用）
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_file = open(os.path.join(RESULT_DIR, "console_predict.txt"), "w", encoding="utf-8")
    stdout_bak = sys.stdout

    class _Tee:
        def write(self, text):
            stdout_bak.write(text)
            log_file.write(text)
            log_file.flush()

        def flush(self):
            stdout_bak.flush()
            log_file.flush()

        def isatty(self):
            return False

    sys.stdout = _Tee()
    try:
        model, class_names, tf, ckpt = load_model(args.model)
        print(f"已加载模型 weights/{args.model}_best.pth | 设备 {DEVICE} | "
              f"验证准确率 {ckpt.get('val_acc', 0):.2%} | 类别 {class_names}")

        if args.demo:
            command_demo(model, class_names, tf)
        elif args.image:
            pred, conf, probs = predict_image(model, class_names, tf, args.image)
            print_result(args.image, pred, conf, probs, class_names)
            plot_single(args.image, pred, conf, probs, class_names)
        elif args.folder:
            command_folder(model, class_names, tf, args.folder)
        else:
            interactive(model, class_names, tf)
    finally:
        sys.stdout = stdout_bak
        log_file.close()


if __name__ == "__main__":
    main()
