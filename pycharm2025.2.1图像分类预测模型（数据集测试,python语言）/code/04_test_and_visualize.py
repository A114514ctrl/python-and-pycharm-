# -*- coding: utf-8 -*-
"""
04_test_and_visualize.py   测试集评估 + 预测结果可视化 + 修改前后对比
=======================================================================
用法（在 code 目录下运行）：
    python 04_test_and_visualize.py --model improved     # 测试改进网络（默认）
    python 04_test_and_visualize.py --model baseline     # 测试基线网络
    python 04_test_and_visualize.py --compare            # 生成“修改前后对比图”

本脚本输出（都放在 results/ 目录）：
    metrics_<模型名>.json          总体准确率 / 每类准确率 / 精确率 / 召回率 / F1
    confusion_matrix_<模型名>.png  混淆矩阵热力图
    prediction_grid_<模型名>.png   预测结果可视化
                                   第一行：测试图片
                                   第二行：预测类别 + 预测置信度
                                   第三行：真实类别
    console_test_<模型名>.txt      测试过程控制台输出（作业截图来源）
    comparison.png（仅 --compare）  修改前 vs 修改后：准确率、曲线、参数量对比
=======================================================================
"""

import argparse
import json
import os
import random
import sys
from importlib import import_module

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
models_mod = import_module("02_models")
build_model, count_parameters = models_mod.build_model, models_mod.count_parameters

PROJECT = os.path.dirname(HERE)
DATASET_DIR = os.path.join(PROJECT, "dataset")
WEIGHT_DIR = os.path.join(PROJECT, "weights")
RESULT_DIR = os.path.join(PROJECT, "results")
CLASS_CN = {"airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫", "deer": "鹿",
            "dog": "狗", "frog": "青蛙", "horse": "马", "ship": "轮船", "truck": "卡车"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    """控制台输出同时写入日志文件。"""

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


def load_model(model_name, weight_path=None):
    """加载训练好的权重。"""
    weight_path = weight_path or os.path.join(WEIGHT_DIR, f"{model_name}_best.pth")
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    class_names = ckpt["classes"]
    model = build_model(ckpt.get("model_name", model_name), num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, ckpt, weight_path


def build_eval_transform(mean, std):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def evaluate(model, class_names, mean, std, batch_size=256, workers=4):
    """在整个测试集上评估，返回全部预测信息。"""
    test_set = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),
                                    transform=build_eval_transform(mean, std))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    all_pred, all_true, all_conf = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probs = torch.softmax(logits.float(), dim=1).cpu()
            conf, pred = probs.max(1)
            all_pred.extend(pred.tolist())
            all_true.extend(labels.tolist())
            all_conf.extend(conf.tolist())

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)                 # 行 = 真实，列 = 预测
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1
    acc = float((np.array(all_true) == np.array(all_pred)).mean())
    return test_set, np.array(all_true), np.array(all_pred), np.array(all_conf), cm, acc


def per_class_metrics(cm, class_names):
    """由混淆矩阵算每类精确率 / 召回率 / F1。"""
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        support = cm[i].sum()
        pred_total = cm[:, i].sum()
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": name, "class_cn": CLASS_CN.get(name, name), "support": int(support),
                     "correct": int(tp), "precision": precision, "recall": recall, "f1": f1})
    return rows


def plot_confusion(cm, class_names, model_name, acc):
    """混淆矩阵热力图（左：数量，右：按行归一化）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    ticks = [f"{n}\n{CLASS_CN.get(n, '')}" for n in class_names]

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"混淆矩阵（数量） accuracy={acc:.2%}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm[i, j]
            axes[0].text(j, i, str(v), ha="center", va="center", fontsize=7,
                         color="white" if v > cm.max() * 0.5 else "black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im1 = axes[1].imshow(cm_norm, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("混淆矩阵（按真实类别归一化）")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm_norm[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if v > 0.5 else "black")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xticks(range(len(class_names)), ticks, fontsize=8, rotation=45)
        ax.set_yticks(range(len(class_names)), ticks, fontsize=8)
        ax.set_xlabel("预测类别 Predict")
        ax.set_ylabel("真实类别 Label")
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[OK] 混淆矩阵已保存：{path}")


def plot_prediction_grid(test_set, true_idx, pred_idx, conf, class_names, model_name,
                         acc, cols=8, seed=0):
    """预测结果可视化：第一行图片、第二行预测类别+置信度、第三行真实类别。"""
    rng = random.Random(seed)
    correct = [i for i in range(len(true_idx)) if true_idx[i] == pred_idx[i]]
    wrong = [i for i in range(len(true_idx)) if true_idx[i] != pred_idx[i]]
    n_wrong = min(2, len(wrong), cols - 1)
    picks = rng.sample(correct, cols - n_wrong) + rng.sample(wrong, n_wrong)
    rng.shuffle(picks)

    fig, axes = plt.subplots(3, cols, figsize=(cols * 1.62, 4.6),
                             gridspec_kw={"height_ratios": [2.4, 1.0, 1.0]})
    for c, idx in enumerate(picks):
        path, _ = test_set.samples[idx]
        img = Image.open(path)
        t, p, cf = true_idx[idx], pred_idx[idx], conf[idx]
        ok = (t == p)

        axes[0][c].imshow(img)
        axes[0][c].set_xticks([]), axes[0][c].set_yticks([])
        for s in axes[0][c].spines.values():
            s.set_color("green" if ok else "red")
            s.set_linewidth(2)

        axes[1][c].axis("off")
        axes[1][c].text(0.5, 0.5, f"预测: {class_names[p]}\n置信度 {cf:.2%}",
                        ha="center", va="center", fontsize=9.5,
                        color="green" if ok else "red", weight="bold",
                        transform=axes[1][c].transAxes)

        axes[2][c].axis("off")
        axes[2][c].text(0.5, 0.5, f"真实: {class_names[t]}\n({CLASS_CN.get(class_names[t], '')})",
                        ha="center", va="center", fontsize=9.5, color="black",
                        transform=axes[2][c].transAxes)

    axes[0][0].set_ylabel("测试图片", fontsize=10)
    axes[1][0].set_ylabel("预测类别\n+置信度", fontsize=10)
    axes[2][0].set_ylabel("真实类别", fontsize=10)
    fig.suptitle(f"{model_name} 模型在测试集上的预测结果（总体准确率 {acc:.2%}，"
                 f"绿框=预测正确，红框=预测错误）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(RESULT_DIR, f"prediction_grid_{model_name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[OK] 预测结果可视化图已保存：{path}")


def test_model(model_name, save_grid=True):
    """评估一个模型，打印结果并保存图表和指标。"""
    model, class_names, ckpt, weight_path = load_model(model_name)
    mean, std = ckpt.get("mean", [0.5] * 3), ckpt.get("std", [0.5] * 3)
    total, trainable, head = count_parameters(model)

    include_test = bool(ckpt.get("include_test", False))
    print("=" * 80)
    print(f"测试模型 : {model_name}    权重文件 : {os.path.relpath(weight_path, PROJECT)}")
    print(f"权重来源 : 第 {ckpt.get('epoch')} 个 epoch 保存（验证准确率 {ckpt.get('val_acc', 0):.2%}）")
    print(f"训练数据 : {'train + test（测试集已参与训练）' if include_test else 'train（测试集不参与训练）'}"
          f"    训练集规模 : {ckpt.get('train_size', '未知')} 张")
    print(f"参数量   : 总 {total:,} / 分类头 {head:,}    设备 : {DEVICE}")
    print("-" * 80)

    test_set, true_idx, pred_idx, conf, cm, acc = evaluate(model, class_names, mean, std)
    rows = per_class_metrics(cm, class_names)

    print(f"测试集图片总数 : {len(test_set)} 张"
          f"（{len(class_names)} 类，每类 {len(test_set) // max(len(class_names), 1)} 张）")
    print(f"总体准确率     : {acc:.2%}   "
          f"（作业要求 >= 85%：{'达标' if acc >= 0.85 else '未达标'}）")
    if include_test:
        print(f"提示           : 按新要求测试集已并入训练，该数值反映“对已见数据的拟合效果”；"
              f"泛化能力请以独立验证集为准（{ckpt.get('val_acc', 0):.2%}）")
    print("-" * 80)
    print(f"{'类别':<12}{'中文':<8}{'测试张数':<10}{'预测正确':<10}{'精确率':<10}{'召回率':<10}{'F1':<10}")
    for r in rows:
        print(f"{r['class']:<12}{r['class_cn']:<8}{r['support']:<10}{r['correct']:<10}"
              f"{r['precision']:<10.4f}{r['recall']:<10.4f}{r['f1']:<10.4f}")
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    top_conf = float(conf.mean())
    print("-" * 80)
    print(f"宏平均 F1 = {macro_f1:.4f}    平均预测置信度 = {top_conf:.4f}")
    print("=" * 80)

    metrics = {
        "model": model_name, "params": total, "head_params": head,
        "test_total": int(len(test_set)), "test_acc": acc, "macro_f1": macro_f1,
        "mean_confidence": top_conf, "val_acc": ckpt.get("val_acc", 0.0),
        "best_epoch": ckpt.get("epoch", 0), "per_class": rows,
        "confusion_matrix": cm.tolist(), "class_names": class_names,
        "include_test": include_test, "train_size": ckpt.get("train_size"),
    }
    with open(os.path.join(RESULT_DIR, f"metrics_{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_confusion(cm, class_names, model_name, acc)
    if save_grid:
        plot_prediction_grid(test_set, true_idx, pred_idx, conf, class_names, model_name, acc)
    return metrics


def plot_comparison():
    """修改前 vs 修改后：准确率柱状图 + 验证曲线 + 参数量对比。"""
    hist_b = json.load(open(os.path.join(RESULT_DIR, "history_baseline.json"), encoding="utf-8"))
    hist_i = json.load(open(os.path.join(RESULT_DIR, "history_improved.json"), encoding="utf-8"))
    met_b = json.load(open(os.path.join(RESULT_DIR, "metrics_baseline.json"), encoding="utf-8"))
    met_i = json.load(open(os.path.join(RESULT_DIR, "metrics_improved.json"), encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (1) 测试准确率柱状图
    names = ["修改前\nBaselineCNN", "修改后\nImprovedCNN"]
    accs = [met_b["test_acc"], met_i["test_acc"]]
    bars = axes[0].bar(names, accs, color=["#9bb7d4", "#2e75b6"], width=0.55)
    for b, a in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width() / 2, a + 0.012, f"{a:.2%}",
                     ha="center", fontsize=11, weight="bold")
    axes[0].axhline(0.85, ls="--", c="r", lw=1.2, label="要求 85%")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("测试集准确率")
    axes[0].set_title("测试准确率对比")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # (2) 验证准确率曲线对比
    ep_b = range(1, len(hist_b["val_acc"]) + 1)
    ep_i = range(1, len(hist_i["val_acc"]) + 1)
    axes[1].plot(ep_b, hist_b["val_acc"], "-", color="#9bb7d4", label=f"修改前（{hist_b['epochs']} epochs）")
    axes[1].plot(ep_i, hist_i["val_acc"], "-", color="#2e75b6", label=f"修改后（{hist_i['epochs']} epochs）")
    axes[1].axhline(0.85, ls="--", c="r", lw=1.2, label="要求 85%")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("验证集准确率")
    axes[1].set_title("验证准确率收敛曲线对比")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (3) 参数量 / F1 / 每类准确率对比
    labels = [f"{n}\n{CLASS_CN.get(n, '')}" for n in met_i["class_names"]]
    acc_b = [r["recall"] for r in met_b["per_class"]]
    acc_i = [r["recall"] for r in met_i["per_class"]]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.19, acc_b, 0.38, label="修改前", color="#9bb7d4")
    axes[2].bar(x + 0.19, acc_i, 0.38, label="修改后", color="#2e75b6")
    axes[2].set_xticks(x, labels, fontsize=7.5, rotation=45)
    axes[2].set_ylabel("每类召回率（该类的准确率）")
    axes[2].set_title(f"每类准确率对比  参数量 {met_b['params'] / 1e6:.2f}M → {met_i['params'] / 1e6:.2f}M")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("网络模型修改前后效果对比（同一数据集、同一增强策略、同一优化器）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(RESULT_DIR, "comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[OK] 修改前后对比图已保存：{path}")


def main():
    parser = argparse.ArgumentParser(description="测试集评估与可视化")
    parser.add_argument("--model", type=str, default="improved", choices=["baseline", "improved"])
    parser.add_argument("--compare", action="store_true", help="生成修改前后对比图")
    parser.add_argument("--no-grid", action="store_true", help="不画预测结果可视化图")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULT_DIR, f"console_test_{args.model}.txt"))
    sys.stdout = tee
    try:
        if args.model:
            test_model(args.model, save_grid=not args.no_grid)
        if args.compare:
            plot_comparison()
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
