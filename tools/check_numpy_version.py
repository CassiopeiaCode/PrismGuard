#!/usr/bin/env python3
"""
检查 NumPy 版本和 fastText 兼容性

使用方法:
    python tools/check_numpy_version.py
"""

import sys

def check_numpy_version():
    """检查 NumPy 版本"""
    print("="*60)
    print("NumPy 版本检查")
    print("="*60)
    
    try:
        import numpy as np
        version = np.__version__
        major, minor = map(int, version.split('.')[:2])
        
        print(f"\n✅ NumPy 已安装")
        print(f"   版本: {version}")
        
        if major >= 2:
            print(f"\n⚠️  警告: 检测到 NumPy 2.x 版本!")
            print(f"   fastText 库尚未完全兼容 NumPy 2.0")
            print(f"\n建议操作:")
            print(f"   1. 降级 NumPy: pip install 'numpy<2.0'")
            print(f"   2. 或切换到 BoW 模型（不依赖 NumPy）")
            return False
        else:
            print(f"\n✅ NumPy 版本兼容 (< 2.0)")
            return True
            
    except ImportError:
        print(f"\n❌ NumPy 未安装")
        print(f"   请安装: pip install 'numpy<2.0'")
        return False


def check_fasttext():
    """检查 fastText 安装"""
    print(f"\n{'='*60}")
    print("fastText 检查")
    print("="*60)
    
    try:
        import fasttext
        print(f"\n✅ fastText 已安装")
        
        # 尝试简单的训练测试
        print(f"\n测试 fastText 基本功能...")
        import tempfile
        import os
        
        # 创建临时训练文件
        fd, train_file = tempfile.mkstemp(suffix=".txt", prefix="fasttext_test_")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write("__label__0 这是正常文本\n")
                f.write("__label__1 这是违规文本\n")
                f.write("__label__0 另一个正常文本\n")
                f.write("__label__1 另一个违规文本\n")
            
            # 尝试训练
            print(f"   训练测试模型...")
            model = fasttext.train_supervised(
                input=train_file,
                dim=10,
                lr=0.1,
                epoch=1,
                wordNgrams=1,
                verbose=0
            )
            
            # 尝试预测
            print(f"   测试预测...")
            labels, probs = model.predict("测试文本", k=2)
            
            print(f"\n✅ fastText 功能正常")
            print(f"   预测标签: {labels}")
            print(f"   预测概率: {probs}")
            return True
            
        except Exception as e:
            print(f"\n❌ fastText 功能测试失败!")
            print(f"   错误: {e}")
            
            # 检查是否是 NumPy 兼容性问题
            error_msg = str(e)
            if "copy" in error_msg and "array" in error_msg:
                print(f"\n⚠️  这是 NumPy 2.0 兼容性问题!")
                print(f"   请降级 NumPy: pip install 'numpy<2.0'")
            
            return False
            
        finally:
            # 清理临时文件
            if os.path.exists(train_file):
                os.remove(train_file)
                
    except ImportError:
        print(f"\n❌ fastText 未安装")
        print(f"   请安装: pip install fasttext-wheel")
        return False


def main():
    print("\n" + "="*60)
    print("环境兼容性检查")
    print("="*60 + "\n")
    
    numpy_ok = check_numpy_version()
    fasttext_ok = check_fasttext()
    
    print(f"\n{'='*60}")
    print("检查结果")
    print("="*60)
    
    if numpy_ok and fasttext_ok:
        print(f"\n✅ 环境配置正常，可以使用 fastText 模型")
    else:
        print(f"\n❌ 环境配置有问题，需要修复:")
        if not numpy_ok:
            print(f"   - NumPy 版本不兼容")
        if not fasttext_ok:
            print(f"   - fastText 功能异常")
        
        print(f"\n推荐解决方案:")
        print(f"   pip install 'numpy<2.0' --force-reinstall")
        print(f"   pip install fasttext-wheel --force-reinstall")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)