#!/usr/bin/env python3
"""
fastText 模型测试工具
用法: python tools/test_fasttext_model.py <profile_name> [text]
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.fasttext_model import fasttext_predict_proba, fasttext_model_exists


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/test_fasttext_model.py <profile_name> [text]")
        print("示例: python tools/test_fasttext_model.py default \"测试文本\"")
        print("\n如果不提供文本，将进入交互模式")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    
    print(f"{'='*60}")
    print(f"fastText 模型测试工具")
    print(f"配置: {profile_name}")
    print(f"{'='*60}\n")
    
    # 加载配置
    profile = ModerationProfile(profile_name)
    
    # 检查模型
    if not fasttext_model_exists(profile):
        print(f"❌ 模型不存在: {profile.get_fasttext_model_path()}")
        print(f"请先运行: python tools/train_fasttext_model.py {profile_name}")
        sys.exit(1)
    
    # 显示阈值配置
    cfg = profile.config.probability
    print(f"阈值配置:")
    print(f"  低风险阈值: {cfg.low_risk_threshold}")
    print(f"  高风险阈值: {cfg.high_risk_threshold}")
    print()
    
    # 单次测试或交互模式
    if len(sys.argv) >= 3:
        # 单次测试
        text = " ".join(sys.argv[2:])
        test_text(text, profile)
    else:
        # 交互模式
        print("进入交互模式，输入文本进行测试（输入 'quit' 退出）\n")
        while True:
            try:
                text = input("请输入文本: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                test_text(text, profile)
                print()
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except EOFError:
                break


def test_text(text: str, profile: ModerationProfile):
    """测试单条文本"""
    print(f"测试文本: {text}")
    print(f"-" * 60)
    
    try:
        # 预测
        prob = fasttext_predict_proba(text, profile)
        
        # 判断结果
        cfg = profile.config.probability
        if prob < cfg.low_risk_threshold:
            result = "✅ 通过（低风险）"
            decision = "直接放行"
        elif prob > cfg.high_risk_threshold:
            result = "❌ 违规（高风险）"
            decision = "直接拒绝"
        else:
            result = "⚠️  不确定"
            decision = "需要 AI 复核"
        
        print(f"违规概率: {prob:.3f}")
        print(f"预测结果: {result}")
        print(f"决策: {decision}")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()