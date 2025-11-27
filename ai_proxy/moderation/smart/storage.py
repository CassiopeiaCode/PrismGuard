"""
审核历史数据存储 - SQLite
"""
import sqlite3
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class Sample(BaseModel):
    """审核样本"""
    id: Optional[int] = None
    text: str
    label: int  # 0=pass, 1=violation
    category: Optional[str] = None
    created_at: Optional[str] = None


class SampleStorage:
    """样本存储管理"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label INTEGER NOT NULL,
                category TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def save_sample(self, text: str, label: int, category: Optional[str] = None):
        """保存样本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO samples (text, label, category) VALUES (?, ?, ?)",
            (text, label, category)
        )
        conn.commit()
        conn.close()
    
    def load_samples(self, max_samples: int = 20000) -> List[Sample]:
        """加载最新的样本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, text, label, category, created_at 
            FROM samples 
            ORDER BY created_at DESC 
            LIMIT ?
            """,
            (max_samples,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4]
            )
            for row in rows
        ]
    
    def get_sample_count(self) -> int:
        """获取样本总数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM samples")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def find_by_text(self, text: str) -> Optional[Sample]:
        """根据文本查找样本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text, label, category, created_at FROM samples WHERE text = ? ORDER BY created_at DESC LIMIT 1",
            (text,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4]
            )
        return None