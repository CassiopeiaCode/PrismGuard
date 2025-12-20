#!/usr/bin/env python3
"""
搜索违规提示词工具

根据关键词搜索数据库中第一个包含该关键词的违规样本的全文
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_proxy.moderation.smart.profile import get_profile
from ai_proxy.moderation.smart.storage import SampleStorage


def search_violation_by_keyword(profile_name: str, keyword: str, max_check: int = 1000):
    """
    搜索包含关键词的第一个违规样本
    
    Args:
        profile_name: 配置文件名（如 'default'）
        keyword: 搜索关键词
        max_check: 最多检查多少条记录
    """
    print(f"\n{'='*60}")
    print(f"搜索违规样本工具")
    print(f"{'='*60}")
    print(f"配置文件: {profile_name}")
    print(f"搜索关键词: {keyword}")
    print(f"最多检查: {max_check} 条记录")
    print(f"{'='*60}\n")
    
    # 加载配置
    try:
        profile = get_profile(profile_name)
        print(f"✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return
    
    # 连接数据库
    db_path = profile.get_db_path()
    if not os.path.exists(db_path):
        print(f"✗ 数据库不存在: {db_path}")
        return
    
    print(f"✓ 数据库路径: {db_path}")
    storage = SampleStorage(db_path)
    
    # 获取样本总数
    total_count = storage.get_sample_count()
    print(f"✓ 数据库总样本数: {total_count}")
    
    # 获取违规样本数量
    pass_count, violation_count = storage.get_label_counts()
    print(f"✓ 样本分布: 正常={pass_count}, 违规={violation_count}")
    
    if violation_count == 0:
        print(f"\n✗ 数据库中没有违规样本")
        return
    
    print(f"\n开始搜索...")
    print(f"{'-'*60}")
    
    # 获取样本ID列表（按时间降序）
    check_limit = min(max_check, total_count)
    sample_ids = storage.get_sample_ids(check_limit)
    
    # 批量加载样本
    samples = storage.load_by_ids(sample_ids)
    
    # 搜索包含关键词的违规样本
    found_count = 0
    checked_count = 0
    
    for sample in samples:
        checked_count += 1
        
        # 只检查违规样本
        if sample.label != 1:
            continue
        
        # 检查是否包含关键词
        if keyword.lower() in sample.text.lower():
            found_count += 1
            print(f"\n{'='*60}")
            print(f"找到第 {found_count} 个匹配的违规样本")
            print(f"{'='*60}")
            print(f"ID: {sample.id}")
            print(f"类别: {sample.category or '未分类'}")
            print(f"创建时间: {sample.created_at}")
            print(f"文本长度: {len(sample.text)} 字符")
            print(f"{'-'*60}")
            print(f"完整文本:")
            print(f"{'-'*60}")
            print(sample.text)
            print(f"{'='*60}")
            
            # 只返回第一个匹配的
            break
    
    print(f"\n搜索完成:")
    print(f"  检查样本数: {checked_count}/{check_limit}")
    print(f"  找到匹配数: {found_count}")
    
    if found_count == 0:
        print(f"\n✗ 未找到包含关键词 '{keyword}' 的违规样本")
        print(f"提示: 可以尝试:")
        print(f"  1. 使用更短或更通用的关键词")
        print(f"  2. 增加 --max-check 参数检查更多记录")
        print(f"  3. 使用 tools/export_violations.py 导出所有违规样本查看")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='搜索数据库中包含特定关键词的违规样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 搜索包含 "SQLite" 的违规样本
  python tools/search_violation_by_keyword.py default SQLite
  
  # 搜索包含 "database" 的违规样本（检查最多2000条）
  python tools/search_violation_by_keyword.py default database --max-check 2000
  
  # 搜索包含 "错误" 的违规样本
  python tools/search_violation_by_keyword.py default 错误
        """
    )
    
    parser.add_argument(
        'profile',
        help='配置文件名（如 default）'
    )
    
    parser.add_argument(
        'keyword',
        help='搜索关键词（不区分大小写）'
    )
    
    parser.add_argument(
        '--max-check',
        type=int,
        default=1000,
        help='最多检查多少条记录（默认: 1000）'
    )
    
    args = parser.parse_args()
    
    search_violation_by_keyword(args.profile, args.keyword, args.max_check)


if __name__ == "__main__":
    main()