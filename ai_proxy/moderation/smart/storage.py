"""
审核历史数据存储 - SQLite with Connection Pool
"""
import sqlite3
import threading
import random
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
from contextlib import contextmanager


class Sample(BaseModel):
    """审核样本"""
    id: Optional[int] = None
    text: str
    label: int  # 0=pass, 1=violation
    category: Optional[str] = None
    created_at: Optional[str] = None


class ConnectionPool:
    """SQLite 连接池"""
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = []
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
    
    @contextmanager
    def get_connection(self):
        """获取连接（上下文管理器）"""
        conn = None
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        try:
            yield conn
        finally:
            with self._lock:
                if len(self._pool) < self.max_connections:
                    self._pool.append(conn)
                else:
                    conn.close()
    
    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()


# 全局连接池字典（每个数据库一个池）
_connection_pools: Dict[str, ConnectionPool] = {}
_pool_lock = threading.Lock()


def get_pool(db_path: str) -> ConnectionPool:
    """获取或创建连接池"""
    with _pool_lock:
        if db_path not in _connection_pools:
            _connection_pools[db_path] = ConnectionPool(db_path)
        return _connection_pools[db_path]


def cleanup_pools():
    """清理所有连接池（应用关闭时调用）"""
    with _pool_lock:
        for pool in _connection_pools.values():
            pool.close_all()
        _connection_pools.clear()


class SampleStorage:
    """样本存储管理"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = get_pool(db_path)
    
    def save_sample(self, text: str, label: int, category: Optional[str] = None):
        """保存样本"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO samples (text, label, category) VALUES (?, ?, ?)",
                (text, label, category)
            )
            conn.commit()
    
    def load_samples(self, max_samples: int = 20000) -> List[Sample]:
        """加载最新的样本"""
        with self.pool.get_connection() as conn:
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
    
    def load_balanced_samples(self, max_samples: int = 20000) -> List[Sample]:
        """
        加载样本并强制保持标签平衡
        
        策略：
        1. 每个标签加载 max_samples/2 个样本
        2. 如果某个标签样本不足，通过复制现有样本来补足
        3. 最终返回完全平衡的样本集（正常:违规 = 1:1）
        
        Args:
            max_samples: 最大样本数（必须是偶数，如果是奇数会向下取整）
            
        Returns:
            平衡的样本列表
        """
        if max_samples <= 0:
            return []
        
        # 确保每个标签的目标数量
        target_per_label = max_samples // 2
        
        if target_per_label == 0:
            return []
        
        pass_count, violation_count = self.get_label_counts()
        
        # 如果某个标签完全没有样本，返回空列表
        if pass_count == 0 or violation_count == 0:
            print(f"[WARNING] 标签不平衡: 正常={pass_count}, 违规={violation_count}")
            print(f"[WARNING] 无法进行平衡采样，返回空列表")
            return []
        
        # 加载正常样本
        pass_samples = self._load_samples_by_label(0, min(target_per_label, pass_count))
        print(f"[BalancedSampling] 正常样本: 加载 {len(pass_samples)}/{target_per_label}")
        
        # 加载违规样本
        violation_samples = self._load_samples_by_label(1, min(target_per_label, violation_count))
        print(f"[BalancedSampling] 违规样本: 加载 {len(violation_samples)}/{target_per_label}")
        
        # 如果正常样本不足，通过复制来补足
        if len(pass_samples) < target_per_label:
            original_count = len(pass_samples)
            while len(pass_samples) < target_per_label:
                # 循环复制样本
                for sample in pass_samples[:original_count]:
                    if len(pass_samples) >= target_per_label:
                        break
                    pass_samples.append(sample)
            print(f"[BalancedSampling] 正常样本不足，已复制到 {len(pass_samples)} 个")
        
        # 如果违规样本不足，通过复制来补足
        if len(violation_samples) < target_per_label:
            original_count = len(violation_samples)
            while len(violation_samples) < target_per_label:
                # 循环复制样本
                for sample in violation_samples[:original_count]:
                    if len(violation_samples) >= target_per_label:
                        break
                    violation_samples.append(sample)
            print(f"[BalancedSampling] 违规样本不足，已复制到 {len(violation_samples)} 个")
        
        # 合并样本
        combined = pass_samples + violation_samples
        
        # 打乱顺序（重要：避免模型学到标签顺序）
        import random
        random.shuffle(combined)
        
        print(f"[BalancedSampling] 最终样本: 正常={len(pass_samples)}, 违规={len(violation_samples)}, 总计={len(combined)}")
        
        return combined
    
    def get_sample_count(self) -> int:
        """获取样本总数"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM samples")
            count = cursor.fetchone()[0]
        return count
    
    def get_sample_ids(self, limit: int) -> List[int]:
        """
        获取样本ID列表（按创建时间降序）
        
        Args:
            limit: 最多返回多少条ID
            
        Returns:
            样本ID列表
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM samples ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    def load_by_ids(self, ids: List[int]) -> List[Sample]:
        """
        根据ID列表批量加载样本
        
        Args:
            ids: 样本ID列表
            
        Returns:
            样本列表（保持ID顺序）
        """
        if not ids:
            return []
        
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            # 使用 IN 查询，然后手动排序以保持输入顺序
            placeholders = ','.join('?' * len(ids))
            query = f"""
                SELECT id, text, label, category, created_at
                FROM samples
                WHERE id IN ({placeholders})
            """
            cursor.execute(query, ids)
            rows = cursor.fetchall()
        
        # 构建样本字典
        samples_dict = {}
        for row in rows:
            samples_dict[row[0]] = Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4]
            )
        
        # 按输入ID顺序返回
        return [samples_dict[id] for id in ids if id in samples_dict]
    
    def find_by_text(self, text: str) -> Optional[Sample]:
        """根据文本查找样本"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, text, label, category, created_at FROM samples WHERE text = ? ORDER BY created_at DESC LIMIT 1",
                (text,)
            )
            row = cursor.fetchone()
        
        if row:
            return Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4]
            )
        return None
    
    def get_label_counts(self) -> Tuple[int, int]:
        """
        获取各标签的样本数量
        
        Returns:
            (pass_count, violation_count) - (成功数, 失败数)
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT label, COUNT(*) FROM samples GROUP BY label"
            )
            rows = cursor.fetchall()
        
        pass_count = 0
        violation_count = 0
        for label, count in rows:
            if label == 0:
                pass_count = count
            elif label == 1:
                violation_count = count
        
        return pass_count, violation_count
    
    def cleanup_excess_samples(self, max_items: int):
        """
        清理超出限制的样本数据
        
        策略:
        1. 如果成功样本数 > max_items/2，随机删除到 max_items/2
        2. 如果失败样本数 > max_items/2，随机删除到 max_items/2
        3. 删除后执行 VACUUM 释放空间
        
        Args:
            max_items: 数据库最大项目数
        """
        total = self.get_sample_count()
        if total <= max_items:
            print(f"[DB清理] 总样本数 {total} <= {max_items}，无需清理")
            return
        
        pass_count, violation_count = self.get_label_counts()
        print(f"[DB清理] 当前样本分布: 成功={pass_count}, 失败={violation_count}, 总计={total}")
        
        target_per_label = max_items // 2
        deleted_count = 0
        
        # 清理成功样本
        if pass_count > target_per_label:
            excess = pass_count - target_per_label
            print(f"[DB清理] 成功样本超出限制 ({pass_count} > {target_per_label})，需删除 {excess} 条")
            deleted = self._delete_random_samples(label=0, count=excess)
            deleted_count += deleted
            print(f"[DB清理] 已删除 {deleted} 条成功样本")
        
        # 清理失败样本
        if violation_count > target_per_label:
            excess = violation_count - target_per_label
            print(f"[DB清理] 失败样本超出限制 ({violation_count} > {target_per_label})，需删除 {excess} 条")
            deleted = self._delete_random_samples(label=1, count=excess)
            deleted_count += deleted
            print(f"[DB清理] 已删除 {deleted} 条失败样本")
        
        if deleted_count > 0:
            # 释放空间
            print(f"[DB清理] 开始 VACUUM 释放空间...")
            with self.pool.get_connection() as conn:
                conn.execute("VACUUM")
            print(f"[DB清理] VACUUM 完成")
            
            # 最终统计
            new_total = self.get_sample_count()
            new_pass, new_violation = self.get_label_counts()
            print(f"[DB清理] 清理完成: 删除 {deleted_count} 条，剩余 {new_total} 条")
            print(f"[DB清理] 新的样本分布: 成功={new_pass}, 失败={new_violation}")
        else:
            print(f"[DB清理] 无需删除样本")
    
    def _load_samples_by_label(self, label: int, limit: int) -> List[Sample]:
        """按标签加载最新样本"""
        if limit <= 0:
            return []
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, text, label, category, created_at
                FROM samples
                WHERE label = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (label, limit)
            )
            rows = cursor.fetchall()
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
    
    def _delete_random_samples(self, label: int, count: int) -> int:
        """
        随机删除指定标签的样本
        
        Args:
            label: 标签 (0=pass, 1=violation)
            count: 删除数量
            
        Returns:
            实际删除数量
        """
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取指定标签的所有ID
            cursor.execute(
                "SELECT id FROM samples WHERE label = ?",
                (label,)
            )
            ids = [row[0] for row in cursor.fetchall()]
            
            if len(ids) <= count:
                # 如果总数不足，全部删除
                to_delete = ids
            else:
                # 随机选择要删除的ID
                to_delete = random.sample(ids, count)
            
            # 批量删除
            placeholders = ','.join('?' * len(to_delete))
            cursor.execute(
                f"DELETE FROM samples WHERE id IN ({placeholders})",
                to_delete
            )
            conn.commit()
            
            return len(to_delete)
