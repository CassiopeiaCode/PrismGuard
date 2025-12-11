#!/usr/bin/env python3
"""
导出审核配置中的违规文本

支持多种导出格式：
- JSON: 结构化数据，包含完整信息
- CSV: 表格格式，便于 Excel 查看
- TXT: 纯文本格式，每行一条违规文本
"""
import sqlite3
import sys
import json
import csv
from datetime import datetime
from pathlib import Path


def export_violations(db_path: str, output_path: str, format: str = "json", limit: int = None):
    """
    导出违规文本
    
    Args:
        db_path: 数据库路径
        output_path: 输出文件路径
        format: 导出格式 (json/csv/txt)
        limit: 限制导出数量，None 表示全部导出
    """
    # 检查数据库是否存在
    if not Path(db_path).exists():
        print(f"❌ 错误: 数据库文件不存在: {db_path}")
        sys.exit(1)
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询违规样本 (label=1)
    query = """
        SELECT id, text, label, category, created_at 
        FROM samples 
        WHERE label = 1
        ORDER BY created_at DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    records = cursor.fetchall()
    
    if not records:
        print("⚠️  未找到违规记录")
        conn.close()
        return
    
    print(f"📊 找到 {len(records)} 条违规记录")
    
    # 根据格式导出
    if format == "json":
        export_json(records, output_path)
    elif format == "csv":
        export_csv(records, output_path)
    elif format == "txt":
        export_txt(records, output_path)
    else:
        print(f"❌ 不支持的格式: {format}")
        conn.close()
        sys.exit(1)
    
    conn.close()
    print(f"✅ 导出完成: {output_path}")


def export_json(records, output_path: str):
    """导出为 JSON 格式"""
    data = []
    for record in records:
        id, text, label, category, created_at = record
        data.append({
            "id": id,
            "text": text,
            "label": label,
            "category": category,
            "created_at": created_at
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"📄 JSON 格式: {len(data)} 条记录")


def export_csv(records, output_path: str):
    """导出为 CSV 格式"""
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['ID', '文本', '标签', '类别', '创建时间'])
        
        # 写入数据
        for record in records:
            id, text, label, category, created_at = record
            writer.writerow([id, text, label, category or '', created_at])
    
    print(f"📊 CSV 格式: {len(records)} 条记录")


def export_txt(records, output_path: str):
    """导出为纯文本格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            id, text, label, category, created_at = record
            # 每行一条违规文本
            f.write(f"{text}\n")
    
    print(f"📝 TXT 格式: {len(records)} 条记录")


def print_statistics(db_path: str):
    """打印数据库统计信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 总数统计
    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]
    
    # 按标签统计
    cursor.execute("""
        SELECT label, COUNT(*) 
        FROM samples 
        GROUP BY label
    """)
    stats = cursor.fetchall()
    
    print(f"\n📈 数据库统计:")
    print(f"  总记录数: {total}")
    for label, count in stats:
        label_str = "违规" if label == 1 else "通过"
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {label_str}: {count} 条 ({percentage:.1f}%)")
    
    conn.close()


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python export_violations.py <db_path> <output_path> [format] [limit]")
        print()
        print("参数:")
        print("  db_path      - 数据库路径")
        print("  output_path  - 输出文件路径")
        print("  format       - 导出格式: json/csv/txt (默认: json)")
        print("  limit        - 限制导出数量 (默认: 全部)")
        print()
        print("示例:")
        print("  # 导出为 JSON 格式")
        print("  python export_violations.py configs/mod_profiles/default/history.db violations.json")
        print()
        print("  # 导出为 CSV 格式")
        print("  python export_violations.py configs/mod_profiles/default/history.db violations.csv csv")
        print()
        print("  # 导出为 TXT 格式（仅文本内容）")
        print("  python export_violations.py configs/mod_profiles/default/history.db violations.txt txt")
        print()
        print("  # 只导出最近 100 条")
        print("  python export_violations.py configs/mod_profiles/default/history.db violations.json json 100")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output_path = sys.argv[2]
    format = sys.argv[3] if len(sys.argv) > 3 else "json"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # 打印统计信息
    print_statistics(db_path)
    print()
    
    # 导出违规文本
    export_violations(db_path, output_path, format, limit)


if __name__ == "__main__":
    main()