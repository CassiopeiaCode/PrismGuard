#!/usr/bin/env python3
"""
fastText 模型评估脚本

评估当前 fastText 模型在数据库样本上的性能指标（支持随机采样）：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数 (F1 Score)
- 混淆矩阵 (Confusion Matrix)

根据配置自动选择：
- use_jieba=true: 使用 jieba 分词进行预测
- use_jieba=false: 使用原版字符级 n-gram

使用方法:
    python tools/evaluate_fasttext_model.py <profile_name> [--sample-size N]
    
参数:
    profile_name: 配置名称
    --sample-size N: 每个标签最多采样 N 个样本（默认100，设为0表示全量评估）
    
示例:
    python tools/evaluate_fasttext_model.py default
    python tools/evaluate_fasttext_model.py 4claude --sample-size 100
    python tools/evaluate_fasttext_model.py default --sample-size 0  # 全量评估
"""

import sys
import os
import random
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_proxy.moderation.smart.profile import get_profile
from ai_proxy.moderation.smart.storage import SampleStorage
from ai_proxy.moderation.smart.fasttext_model import fasttext_model_exists, _load_fasttext_with_cache
import jieba
from tqdm import tqdm


def evaluate_fasttext_model(profile_name: str, sample_size: int = 100):
    """
    评估 fastText 模型性能
    
    Args:
        profile_name: 配置名称
        sample_size: 每个标签最多采样的样本数（0表示全量）
    """
    print(f"\n{'='*60}")
    print(f"fastText 模型评估")
    print(f"{'='*60}")
    print(f"配置: {profile_name}\n")
    
    # 加载配置
    try:
        profile = get_profile(profile_name)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return
    
    # 检查模型是否存在
    if not fasttext_model_exists(profile):
        print(f"❌ fastText 模型不存在: {profile.get_fasttext_model_path()}")
        print(f"   请先训练模型: python tools/train_fasttext_model.py {profile_name}")
        return
    
    print(f"✅ 模型文件: {profile.get_fasttext_model_path()}")
    
    # 检查是否使用 jieba
    use_jieba = profile.config.fasttext_training.use_jieba
    print(f"✅ 分词方式: {'jieba 中文分词' if use_jieba else '字符级 n-gram'}")
    
    # 加载数据库样本
    storage = SampleStorage(profile.get_db_path())
    total_count = storage.get_sample_count()
    
    if total_count == 0:
        print(f"❌ 数据库中没有样本")
        return
    
    # 获取标签分布
    pass_count, violation_count = storage.get_label_counts()
    print(f"\n数据库样本总数: {total_count}")
    print(f"  正常 (label=0): {pass_count} 条")
    print(f"  违规 (label=1): {violation_count} 条")
    
    # 采样策略
    if sample_size > 0:
        print(f"\n采样策略: 每个标签最多 {sample_size} 个样本")
        
        # 分别加载两个标签的样本
        pass_samples = storage._load_samples_by_label(0, min(sample_size, pass_count))
        violation_samples = storage._load_samples_by_label(1, min(sample_size, violation_count))
        
        # 随机打乱
        random.shuffle(pass_samples)
        random.shuffle(violation_samples)
        
        # 取前N个
        pass_samples = pass_samples[:sample_size]
        violation_samples = violation_samples[:sample_size]
        
        samples = pass_samples + violation_samples
        random.shuffle(samples)  # 打乱顺序
        
        print(f"  实际采样: 正常 {len(pass_samples)} 条, 违规 {len(violation_samples)} 条")
    else:
        print(f"\n全量评估模式")
        samples = storage.load_samples(max_samples=total_count)
    
    if not samples:
        print(f"❌ 没有可用样本")
        return
    
    print(f"✅ 评估样本数: {len(samples)}")
    
    # 加载模型
    print(f"\n加载 fastText 模型...")
    try:
        model = _load_fasttext_with_cache(profile)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✅ 模型加载成功")
    
    # 预测所有样本
    print(f"\n开始预测...")
    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    y_proba = []  # 预测概率
    
    # 使用 tqdm 显示预测进度
    for sample in tqdm(samples, desc="模型预测", unit="样本"):
        # 预处理文本
        text = sample.text.replace('\n', ' ').replace('\r', ' ')
        
        # 根据配置选择分词方式
        if use_jieba:
            # 使用 jieba 分词
            words = jieba.cut(text)
            text = ' '.join(words)
        
        # 预测
        try:
            labels, probs = model.predict(text, k=2)
            
            # 找出违规标签的概率
            violation_prob = 0.0
            for label, p in zip(labels, probs):
                if label == "__label__1":
                    violation_prob = float(p)
                    break
            
            # 使用 0.5 作为阈值判断
            pred_label = 1 if violation_prob >= 0.5 else 0
            
            y_true.append(sample.label)
            y_pred.append(pred_label)
            y_proba.append(violation_prob)
            
        except Exception as e:
            tqdm.write(f"  ⚠️ 预测失败: {e}")
            continue
    
    print(f"\n✅ 预测完成: {len(y_pred)}/{len(samples)} 条成功")
    
    # 计算评估指标
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}\n")
    
    # 混淆矩阵
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)  # 真阳性
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)  # 真阴性
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)  # 假阳性
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)  # 假阴性
    
    print(f"混淆矩阵:")
    print(f"                预测正常    预测违规")
    print(f"  实际正常        {tn:4d}        {fp:4d}")
    print(f"  实际违规        {fn:4d}        {tp:4d}")
    print()
    
    # 准确率 (Accuracy)
    accuracy = (tp + tn) / len(y_pred) if len(y_pred) > 0 else 0
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  正确预测: {tp + tn}/{len(y_pred)}")
    
    # 精确率 (Precision) - 预测为违规的样本中，真正违规的比例
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"\n精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  预测违规中真违规: {tp}/{tp + fp}")
    print(f"  含义: 模型说违规时，有 {precision*100:.1f}% 的概率是对的")
    
    # 召回率 (Recall) - 真实违规样本中，被正确预测的比例
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  真违规被识别: {tp}/{tp + fn}")
    print(f"  含义: 真实违规内容中，有 {recall*100:.1f}% 被模型识别出来")
    
    # F1 分数 (F1 Score) - 精确率和召回率的调和平均
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\nF1 分数 (F1 Score): {f1:.4f}")
    print(f"  精确率和召回率的调和平均")
    
    # 特异度 (Specificity) - 真实正常样本中，被正确预测的比例
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n特异度 (Specificity): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  真正常被识别: {tn}/{tn + fp}")
    print(f"  含义: 真实正常内容中，有 {specificity*100:.1f}% 被正确放行")
    
    # 假阳性率 (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\n假阳性率 (FPR):    {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  正常被误判违规: {fp}/{fp + tn}")
    print(f"  含义: 正常内容中，有 {fpr*100:.1f}% 被误判为违规")
    
    # 假阴性率 (False Negative Rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\n假阴性率 (FNR):    {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"  违规被误判正常: {fn}/{fn + tp}")
    print(f"  含义: 违规内容中，有 {fnr*100:.1f}% 被漏判为正常")
    
    # 概率分布统计
    print(f"\n{'='*60}")
    print(f"预测概率分布")
    print(f"{'='*60}\n")
    
    # 按真实标签分组统计概率
    proba_0 = [p for t, p in zip(y_true, y_proba) if t == 0]  # 正常样本的预测概率
    proba_1 = [p for t, p in zip(y_true, y_proba) if t == 1]  # 违规样本的预测概率
    
    if proba_0:
        print(f"正常样本 (label=0) 的违规概率分布:")
        print(f"  最小值: {min(proba_0):.4f}")
        print(f"  最大值: {max(proba_0):.4f}")
        print(f"  平均值: {sum(proba_0)/len(proba_0):.4f}")
        print(f"  中位数: {sorted(proba_0)[len(proba_0)//2]:.4f}")
    
    if proba_1:
        print(f"\n违规样本 (label=1) 的违规概率分布:")
        print(f"  最小值: {min(proba_1):.4f}")
        print(f"  最大值: {max(proba_1):.4f}")
        print(f"  平均值: {sum(proba_1)/len(proba_1):.4f}")
        print(f"  中位数: {sorted(proba_1)[len(proba_1)//2]:.4f}")
    
    # 阈值分析
    print(f"\n{'='*60}")
    print(f"阈值分析")
    print(f"{'='*60}\n")
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"{'阈值':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print(f"{'-'*60}")
    
    for threshold in thresholds:
        # 使用不同阈值重新计算
        y_pred_t = [1 if p >= threshold else 0 for p in y_proba]
        
        tp_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 1 and p == 1)
        tn_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 0 and p == 0)
        fp_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 0 and p == 1)
        fn_t = sum(1 for t, p in zip(y_true, y_pred_t) if t == 1 and p == 0)
        
        acc_t = (tp_t + tn_t) / len(y_pred_t) if len(y_pred_t) > 0 else 0
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        
        print(f"{threshold:<8.2f} {acc_t:<10.4f} {prec_t:<10.4f} {rec_t:<10.4f} {f1_t:<10.4f}")
    
    print(f"\n{'='*60}")
    print(f"评估完成")
    print(f"{'='*60}\n")
    
    # 建议
    print(f"💡 建议:")
    if accuracy < 0.8:
        print(f"  ⚠️ 准确率较低 ({accuracy*100:.1f}%)，建议:")
        print(f"     - 增加训练样本数量")
        print(f"     - 检查样本质量和标签准确性")
        print(f"     - 调整模型参数 (dim, lr, epoch 等)")
    
    if precision < 0.7:
        print(f"  ⚠️ 精确率较低 ({precision*100:.1f}%)，误报率高:")
        print(f"     - 考虑提高违规判定阈值 (如 0.6 或 0.7)")
        print(f"     - 增加正常样本的训练数据")
    
    if recall < 0.7:
        print(f"  ⚠️ 召回率较低 ({recall*100:.1f}%)，漏报率高:")
        print(f"     - 考虑降低违规判定阈值 (如 0.3 或 0.4)")
        print(f"     - 增加违规样本的训练数据")
    
    if f1 >= 0.8:
        print(f"  ✅ F1 分数良好 ({f1:.3f})，模型性能较好")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='评估 fastText 模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/evaluate_fasttext_model.py default
  python tools/evaluate_fasttext_model.py 4claude --sample-size 100
  python tools/evaluate_fasttext_model.py default --sample-size 0  # 全量评估
        """
    )
    parser.add_argument('profile_name', help='配置名称')
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='每个标签最多采样的样本数（默认100，设为0表示全量评估）'
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_fasttext_model(args.profile_name, args.sample_size)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()