#!/usr/bin/env python3
"""
fastText 训练诊断工具

分析训练数据质量和模型训练问题

使用方法:
    python tools/diagnose_fasttext_training.py <profile_name>
"""

import sys
import os
import tempfile
from pathlib import Path
from collections import Counter

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_proxy.moderation.smart.profile import get_profile
from ai_proxy.moderation.smart.storage import SampleStorage


def diagnose_training_data(profile_name: str):
    """诊断训练数据"""
    print(f"\n{'='*60}")
    print(f"fastText 训练数据诊断")
    print(f"{'='*60}")
    print(f"配置: {profile_name}\n")
    
    # 加载配置
    try:
        profile = get_profile(profile_name)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return
    
    cfg = profile.config.fasttext_training
    storage = SampleStorage(profile.get_db_path())
    
    # 1. 样本统计
    print(f"{'='*60}")
    print(f"1. 样本统计")
    print(f"{'='*60}\n")
    
    total_count = storage.get_sample_count()
    pass_count, violation_count = storage.get_label_counts()
    
    print(f"数据库样本总数: {total_count}")
    print(f"  正常 (label=0): {pass_count} 条 ({pass_count/total_count*100:.1f}%)")
    print(f"  违规 (label=1): {violation_count} 条 ({violation_count/total_count*100:.1f}%)")
    
    if violation_count == 0:
        print(f"\n❌ 致命问题: 没有违规样本！")
        print(f"   模型无法学习违规模式")
        return
    
    # 计算不平衡比例
    imbalance_ratio = pass_count / violation_count if violation_count > 0 else float('inf')
    print(f"\n类别不平衡比例: {imbalance_ratio:.2f}:1 (正常:违规)")
    
    if imbalance_ratio > 10:
        print(f"⚠️  严重不平衡！建议比例 < 5:1")
    elif imbalance_ratio > 5:
        print(f"⚠️  中度不平衡，建议增加违规样本")
    else:
        print(f"✅ 平衡度良好")
    
    # 2. 训练配置检查
    print(f"\n{'='*60}")
    print(f"2. 训练配置")
    print(f"{'='*60}\n")
    
    print(f"最小样本数: {cfg.min_samples}")
    print(f"最大样本数: {cfg.max_samples}")
    print(f"维度: {cfg.dim}")
    print(f"学习率: {cfg.lr}")
    print(f"训练轮数: {cfg.epoch}")
    print(f"词级 n-gram: {cfg.word_ngrams}")
    print(f"字符级 n-gram: [{cfg.minn}, {cfg.maxn}]")
    
    if total_count < cfg.min_samples:
        print(f"\n❌ 样本数不足: {total_count} < {cfg.min_samples}")
        return
    
    # 3. 模拟训练数据加载
    print(f"\n{'='*60}")
    print(f"3. 训练数据加载模拟")
    print(f"{'='*60}\n")
    
    print(f"使用 load_balanced_samples({cfg.max_samples})...")
    samples = storage.load_balanced_samples(cfg.max_samples)
    
    # 统计加载后的标签分布
    label_dist = Counter(s.label for s in samples)
    print(f"\n加载的样本数: {len(samples)}")
    print(f"  正常 (label=0): {label_dist[0]} 条 ({label_dist[0]/len(samples)*100:.1f}%)")
    print(f"  违规 (label=1): {label_dist[1]} 条 ({label_dist[1]/len(samples)*100:.1f}%)")
    
    train_imbalance = label_dist[0] / label_dist[1] if label_dist[1] > 0 else float('inf')
    print(f"\n训练集不平衡比例: {train_imbalance:.2f}:1")
    
    if train_imbalance > 3:
        print(f"⚠️  训练集仍然不平衡！")
        print(f"   建议: 增加违规样本或使用类别权重")
    
    # 4. 文本长度分析
    print(f"\n{'='*60}")
    print(f"4. 文本长度分析")
    print(f"{'='*60}\n")
    
    pass_samples = [s for s in samples if s.label == 0]
    violation_samples = [s for s in samples if s.label == 1]
    
    if pass_samples:
        pass_lengths = [len(s.text) for s in pass_samples]
        print(f"正常样本文本长度:")
        print(f"  最小: {min(pass_lengths)}")
        print(f"  最大: {max(pass_lengths)}")
        print(f"  平均: {sum(pass_lengths)/len(pass_lengths):.1f}")
        print(f"  中位数: {sorted(pass_lengths)[len(pass_lengths)//2]}")
    
    if violation_samples:
        vio_lengths = [len(s.text) for s in violation_samples]
        print(f"\n违规样本文本长度:")
        print(f"  最小: {min(vio_lengths)}")
        print(f"  最大: {max(vio_lengths)}")
        print(f"  平均: {sum(vio_lengths)/len(vio_lengths):.1f}")
        print(f"  中位数: {sorted(vio_lengths)[len(vio_lengths)//2]}")
    
    # 5. 样本质量检查
    print(f"\n{'='*60}")
    print(f"5. 样本质量检查")
    print(f"{'='*60}\n")
    
    # 检查空文本
    empty_count = sum(1 for s in samples if not s.text.strip())
    if empty_count > 0:
        print(f"⚠️  发现 {empty_count} 个空文本样本")
    else:
        print(f"✅ 没有空文本")
    
    # 检查重复文本
    text_counts = Counter(s.text for s in samples)
    duplicates = sum(1 for count in text_counts.values() if count > 1)
    if duplicates > 0:
        print(f"⚠️  发现 {duplicates} 个重复文本")
        # 显示最常见的重复
        most_common = text_counts.most_common(3)
        print(f"   最常见的重复:")
        for text, count in most_common:
            if count > 1:
                print(f"     - 出现 {count} 次: {text[:50]}...")
    else:
        print(f"✅ 没有重复文本")
    
    # 6. 训练文件预览
    print(f"\n{'='*60}")
    print(f"6. 训练文件格式预览")
    print(f"{'='*60}\n")
    
    print(f"fastText 训练文件格式示例（前5条）:\n")
    for i, sample in enumerate(samples[:5]):
        text = sample.text.replace('\n', ' ').replace('\r', ' ')
        print(f"__label__{sample.label} {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # 7. 问题诊断和建议
    print(f"\n{'='*60}")
    print(f"7. 问题诊断和建议")
    print(f"{'='*60}\n")
    
    issues = []
    suggestions = []
    
    # 检查样本不平衡
    if imbalance_ratio > 10:
        issues.append(f"严重的类别不平衡 ({imbalance_ratio:.1f}:1)")
        suggestions.append("增加 ai_review_rate 以积累更多违规样本")
        suggestions.append("考虑使用数据增强技术")
    
    # 检查违规样本数量
    if violation_count < 100:
        issues.append(f"违规样本太少 ({violation_count} 条)")
        suggestions.append("至少需要 200+ 违规样本才能有效训练")
        suggestions.append("提高 ai_review_rate 到 0.5 或更高")
    
    # 检查训练集不平衡
    if train_imbalance > 3:
        issues.append(f"训练集不平衡 ({train_imbalance:.1f}:1)")
        suggestions.append("fastText 训练时添加类别权重")
        suggestions.append("或手动平衡训练数据")
    
    # 检查模型参数
    if cfg.epoch < 10:
        issues.append(f"训练轮数较少 ({cfg.epoch})")
        suggestions.append("建议增加 epoch 到 20-50")
    
    if issues:
        print(f"发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n建议的解决方案:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print(f"✅ 未发现明显问题")
    
    # 8. fastText 特定建议
    print(f"\n{'='*60}")
    print(f"8. fastText 训练优化建议")
    print(f"{'='*60}\n")
    
    print(f"针对当前数据的建议:")
    print(f"  1. 增加训练轮数: epoch=25 (当前 {cfg.epoch})")
    print(f"  2. 调整学习率: lr=0.5 (当前 {cfg.lr})")
    print(f"  3. 使用 loss='hs' (hierarchical softmax) 替代 'ova'")
    print(f"  4. 添加类别权重以处理不平衡")
    print(f"  5. 考虑使用 autotune 自动优化参数")
    
    print(f"\n示例训练命令:")
    print(f"  model = fasttext.train_supervised(")
    print(f"      input=train_file,")
    print(f"      dim={cfg.dim},")
    print(f"      lr=0.5,  # 提高学习率")
    print(f"      epoch=25,  # 增加轮数")
    print(f"      wordNgrams={cfg.word_ngrams},")
    print(f"      minn={cfg.minn},")
    print(f"      maxn={cfg.maxn},")
    print(f"      loss='hs',  # 使用 hierarchical softmax")
    print(f"      verbose=2")
    print(f"  )")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/diagnose_fasttext_training.py <profile_name>")
        print("\n示例:")
        print("  python tools/diagnose_fasttext_training.py default")
        print("  python tools/diagnose_fasttext_training.py 4claudecode")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    
    try:
        diagnose_training_data(profile_name)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()