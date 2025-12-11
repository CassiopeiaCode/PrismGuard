#!/usr/bin/env python3
"""
fastText 模型训练工具
用法: python tools/train_fasttext_model.py <profile_name>
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.fasttext_model import train_fasttext_model
from ai_proxy.moderation.smart.storage import SampleStorage


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/train_fasttext_model.py <profile_name>")
        print("示例: python tools/train_fasttext_model.py default")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    
    print(f"{'='*60}")
    print(f"fastText 模型训练工具")
    print(f"配置: {profile_name}")
    print(f"{'='*60}\n")
    
    # 加载配置
    profile = ModerationProfile(profile_name)
    
    # 显示配置信息
    cfg = profile.config.fasttext_training
    print(f"训练配置:")
    print(f"  最小样本数: {cfg.min_samples}")
    print(f"  最大样本数: {cfg.max_samples}")
    print(f"  维度: {cfg.dim}")
    print(f"  学习率: {cfg.lr}")
    print(f"  训练轮数: {cfg.epoch}")
    print(f"  词级 n-gram: {cfg.word_ngrams}")
    print(f"  字符级 n-gram: [{cfg.minn}, {cfg.maxn}]")
    print()
    
    # 检查样本数据
    storage = SampleStorage(profile.get_db_path())
    sample_count = storage.get_sample_count()
    pass_count, violation_count = storage.get_label_counts()
    
    print(f"样本统计:")
    print(f"  总数: {sample_count}")
    print(f"  通过: {pass_count}")
    print(f"  违规: {violation_count}")
    print()
    
    if sample_count < cfg.min_samples:
        print(f"❌ 样本数不足 {cfg.min_samples}，无法训练")
        sys.exit(1)
    
    # 开始训练
    print(f"开始训练...\n")
    try:
        train_fasttext_model(profile)
        print(f"\n✅ 训练完成")
        print(f"模型已保存: {profile.get_fasttext_model_path()}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()