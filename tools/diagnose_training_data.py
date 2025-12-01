"""
诊断训练数据问题
检查数据库中的样本标签是否正确
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ai_proxy.moderation.smart.storage import SampleStorage

def diagnose(profile_name: str = "4claudecode"):
    """诊断训练数据"""
    db_path = f"configs/mod_profiles/{profile_name}/history.db"
    
    if not os.path.exists(db_path):
        print(f"数据库不存在: {db_path}")
        return
    
    storage = SampleStorage(db_path)
    
    # 获取统计信息
    total = storage.get_sample_count()
    pass_count, violation_count = storage.get_label_counts()
    
    print(f"=== 数据库统计 ===")
    print(f"总样本数: {total}")
    print(f"label=0 (正常): {pass_count} 条 ({pass_count/total*100:.1f}%)")
    print(f"label=1 (违规): {violation_count} 条 ({violation_count/total*100:.1f}%)")
    print()
    
    # 显示一些样本
    print(f"=== 样本示例 ===")
    samples = storage.load_samples(max_samples=20)
    
    for i, sample in enumerate(samples[:10], 1):
        label_str = "违规" if sample.label == 1 else "正常"
        text_preview = sample.text[:80].replace('\n', ' ')
        print(f"\n样本 {i}:")
        print(f"  标签: {label_str} (label={sample.label})")
        print(f"  文本: {text_preview}...")
        print(f"  类别: {sample.category}")
        print(f"  时间: {sample.created_at}")
    
    print(f"\n=== 诊断建议 ===")
    
    # 检查是否标签可能反了
    if violation_count < pass_count * 0.1:  # 违规样本少于10%
        print(f"⚠️  警告: 违规样本占比很低 ({violation_count/total*100:.1f}%)")
        print(f"   如果实际上大部分请求都被拒绝，说明标签可能反了！")
        print(f"   请检查上面的样本，确认 label=0 的文本是否真的是正常的")
    
    if pass_count < violation_count * 0.1:  # 正常样本少于10%
        print(f"⚠️  警告: 正常样本占比很低 ({pass_count/total*100:.1f}%)")
        print(f"   如果实际上大部分请求都能正常通过，说明标签可能反了！")
        print(f"   请检查上面的样本，确认 label=1 的文本是否真的是违规的")

if __name__ == "__main__":
    import sys
    profile = sys.argv[1] if len(sys.argv) > 1 else "4claudecode"
    diagnose(profile)