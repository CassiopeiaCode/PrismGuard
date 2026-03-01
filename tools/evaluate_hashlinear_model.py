#!/usr/bin/env python3
"""
HashLinear 模型评估脚本

评估 HashLinear 模型在数据库样本上的性能指标（支持随机采样）：
- Accuracy / Precision / Recall / F1

使用方法:
    python tools/evaluate_hashlinear_model.py <profile_name> [--sample-size N]
"""

import argparse
import os
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_proxy.moderation.smart.profile import get_profile
from ai_proxy.moderation.smart.storage import SampleStorage
from ai_proxy.moderation.smart.hashlinear_model import hashlinear_model_exists, hashlinear_predict_proba


def evaluate_hashlinear_model(profile_name: str, sample_size: int = 100):
    print(f"\n{'='*60}")
    print("HashLinear 模型评估")
    print(f"{'='*60}")
    print(f"配置: {profile_name}\n")

    try:
        profile = get_profile(profile_name)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return

    if not hashlinear_model_exists(profile):
        print(f"❌ HashLinear 模型不存在: {profile.get_hashlinear_model_path()}")
        print(f"   请先训练模型: python tools/train_hashlinear_model.py {profile_name}")
        return

    print(f"✅ 模型文件: {profile.get_hashlinear_model_path()}")

    storage = SampleStorage(profile.get_db_path(), read_only=True)
    total_count = storage.get_sample_count()
    if total_count == 0:
        print("❌ 数据库中没有样本")
        return

    pass_count, violation_count = storage.get_label_counts()
    print(f"\n数据库样本总数: {total_count}")
    print(f"  正常 (label=0): {pass_count} 条")
    print(f"  违规 (label=1): {violation_count} 条")

    if sample_size > 0:
        print(f"\n采样策略: 每个标签最多 {sample_size} 个样本")
        pass_samples = storage._load_samples_by_label(0, min(sample_size, pass_count))
        violation_samples = storage._load_samples_by_label(1, min(sample_size, violation_count))
        random.shuffle(pass_samples)
        random.shuffle(violation_samples)
        pass_samples = pass_samples[:sample_size]
        violation_samples = violation_samples[:sample_size]
        samples = pass_samples + violation_samples
        random.shuffle(samples)
        print(f"  实际采样: 正常 {len(pass_samples)} 条, 违规 {len(violation_samples)} 条")
    else:
        print("\n全量评估模式")
        samples = storage.load_samples(max_samples=total_count)

    if not samples:
        print("❌ 没有可用样本")
        return

    print(f"✅ 评估样本数: {len(samples)}")

    y_true = []
    y_pred = []
    y_proba = []

    print("\n开始预测...")
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i + 1}/{len(samples)}")
        try:
            p = hashlinear_predict_proba(s.text, profile)
            pred = 1 if p >= 0.5 else 0
            y_true.append(int(s.label))
            y_pred.append(pred)
            y_proba.append(p)
        except Exception as e:
            print(f"  ⚠️ 预测失败 (样本 {i+1}): {e}")

    if not y_pred:
        print("❌ 没有成功预测的样本")
        return

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / len(y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}\n")
    print("混淆矩阵:")
    print("                预测正常    预测违规")
    print(f"  实际正常        {tn:4d}        {fp:4d}")
    print(f"  实际违规        {fn:4d}        {tp:4d}")
    print()
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数 (F1 Score): {f1:.4f}")
    print(f"\n✅ 评估完成: {len(y_pred)}/{len(samples)} 条成功")

    print(f"\n{'='*60}")
    print("阈值分析（细粒度）")
    print(f"{'='*60}\n")

    thresholds = [i / 100 for i in range(5, 100, 5)]  # 0.05 .. 0.95
    print(f"{'阈值':<8} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'FPR':<10} {'FNR':<10}")
    print(f"{'-'*78}")

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred_t = [1 if p >= threshold else 0 for p in y_proba]

        tp_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 1 and p == 1)
        tn_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 0 and p == 0)
        fp_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 0 and p == 1)
        fn_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 1 and p == 0)

        acc_t = (tp_t + tn_t) / len(y_pred_t) if len(y_pred_t) > 0 else 0.0
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0

        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = threshold

        print(
            f"{threshold:<8.2f} "
            f"{acc_t:<12.4f} "
            f"{prec_t:<12.4f} "
            f"{rec_t:<12.4f} "
            f"{f1_t:<12.4f} "
            f"{fpr_t:<10.4f} "
            f"{fnr_t:<10.4f}"
        )

    print(f"\n{'='*80}")
    print(f"🎯 最佳阈值: {best_threshold:.2f} (F1 分数: {best_f1:.4f})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_name")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    evaluate_hashlinear_model(args.profile_name, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
