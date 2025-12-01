"""
修复 BOW 模型训练问题
通过数据增强和调整训练参数
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ai_proxy.moderation.smart.profile import get_profile
from ai_proxy.moderation.smart.storage import SampleStorage
from ai_proxy.moderation.smart.bow import train_bow_model

def fix_model(profile_name: str = "4claudecode"):
    """修复模型"""
    profile = get_profile(profile_name)
    storage = SampleStorage(profile.get_db_path())
    
    # 检查数据
    total = storage.get_sample_count()
    pass_count, violation_count = storage.get_label_counts()
    
    print(f"当前数据统计:")
    print(f"  总样本: {total}")
    print(f"  正常: {pass_count} ({pass_count/total*100:.1f}%)")
    print(f"  违规: {violation_count} ({violation_count/total*100:.1f}%)")
    print()
    
    if violation_count < 10:
        print("❌ 错误：违规样本太少！")
        print("   BOW 模型需要至少 50-100 个违规样本才能有效训练")
        print("   当前只有", violation_count, "个违规样本")
        print()
        print("建议方案：")
        print("1. 收集更多违规样本（等待 AI 审核积累）")
        print("2. 暂时禁用 BOW 模型，全部使用 AI 审核")
        print("3. 提高 AI 审核率到 100%，快速积累标注数据")
        return False
    
    if total < 200:
        print("⚠️  警告：总样本数太少")
        print("   建议至少收集 200 个样本后再训练")
        return False
    
    print("✅ 数据充足，开始训练...")
    train_bow_model(profile)
    return True

if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "4claudecode"
    success = fix_model(profile)
    
    if not success:
        print("\n请修改配置文件，暂时禁用 BOW 模型：")
        print(f"configs/mod_profiles/{profile}/profile.json")
        print("""
{
  "bow_training": {
    "min_samples": 999999  // 设置一个很大的值，禁用自动训练
  }
}
""")