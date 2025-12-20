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
        """初始化数据库表结构并启用 WAL 模式"""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0  # 增加超时时间到 30 秒
        )
        cursor = conn.cursor()
        
        # 启用 WAL 模式以提高并发性能
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # 设置较长的 busy_timeout
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 秒
        
        # 创建表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label INTEGER NOT NULL,
                category TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建文本索引以加速查询
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_text
            ON samples(text)
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
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0,  # 30 秒超时
                    isolation_level=None  # 自动提交模式,减少锁定
                )
                # 为新连接设置 busy_timeout
                conn.execute("PRAGMA busy_timeout=30000")
        
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
        """保存样本（带重试机制）"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                with self.pool.get_connection() as conn:
                    cursor = conn.cursor()
                    # 使用 INSERT OR IGNORE 避免重复插入
                    cursor.execute(
                        "INSERT OR IGNORE INTO samples (text, label, category) VALUES (?, ?, ?)",
                        (text, label, category)
                    )
                    if conn.isolation_level is not None:
                        conn.commit()
                return  # 成功则返回
            
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    print(f"[WARNING] 数据库锁定,重试 {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                raise
    
    def load_samples(self, max_samples: int = 20000) -> List[Sample]:
        """加载最新的样本（按时间倒序）"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, text, label, category, created_at
                FROM samples
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max_samples,),
            )
            rows = cursor.fetchall()

        return [
            Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4],
            )
            for row in rows
        ]

    def load_random_samples(self, max_samples: int = 20000) -> List[Sample]:
        """随机加载样本（不做 1:1 平衡，不复制样本）"""
        if max_samples <= 0:
            return []

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, text, label, category, created_at
                FROM samples
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (max_samples,),
            )
            rows = cursor.fetchall()

        return [
            Sample(
                id=row[0],
                text=row[1],
                label=row[2],
                category=row[3],
                created_at=row[4],
            )
            for row in rows
        ]

    def load_balanced_latest_samples(self, max_samples: int = 20000) -> List[Sample]:
        """
        平衡加载最新样本（每类最多 max_samples/2，按时间倒序，不随机）
        """
        if max_samples <= 0:
            return []

        pass_count, violation_count = self.get_label_counts()
        if pass_count == 0 or violation_count == 0:
            print(f"[WARNING] 标签不平衡: 正常={pass_count}, 违规={violation_count}")
            print(f"[WARNING] 无法进行平衡采样，返回空列表")
            return []

        target_per_label = max_samples // 2
        if target_per_label <= 0:
            return []

        balanced_count = min(pass_count, violation_count, target_per_label)

        print(f"[BalancedLatest] 数据库样本分布: 正常={pass_count}, 违规={violation_count}")
        print(f"[BalancedLatest] 每类取最新 {balanced_count} 个样本（不随机）")

        pass_samples = self._load_samples_by_label(0, balanced_count)
        violation_samples = self._load_samples_by_label(1, balanced_count)

        combined = pass_samples + violation_samples
        random.shuffle(combined)
        print(f"[BalancedLatest] 最终样本: 正常={len(pass_samples)}, 违规={len(violation_samples)}, 总计={len(combined)}")
        return combined

    def load_balanced_random_samples(self, max_samples: int = 20000) -> List[Sample]:
        """
        平衡随机加载样本（每类最多 max_samples/2，全库随机，不复制）
        """
        if max_samples <= 0:
            return []

        pass_count, violation_count = self.get_label_counts()
        if pass_count == 0 or violation_count == 0:
            print(f"[WARNING] 标签不平衡: 正常={pass_count}, 违规={violation_count}")
            print(f"[WARNING] 无法进行平衡采样，返回空列表")
            return []

        target_per_label = max_samples // 2
        if target_per_label <= 0:
            return []

        balanced_count = min(pass_count, violation_count, target_per_label)

        print(f"[BalancedRandom] 数据库样本分布: 正常={pass_count}, 违规={violation_count}")
        print(f"[BalancedRandom] 每类随机抽取 {balanced_count} 个样本（不复制）")

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, text, label, category, created_at
                FROM samples
                WHERE label = 0
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (balanced_count,),
            )
            pass_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT id, text, label, category, created_at
                FROM samples
                WHERE label = 1
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (balanced_count,),
            )
            violation_rows = cursor.fetchall()

        pass_samples = [
            Sample(id=r[0], text=r[1], label=r[2], category=r[3], created_at=r[4])
            for r in pass_rows
        ]
        violation_samples = [
            Sample(id=r[0], text=r[1], label=r[2], category=r[3], created_at=r[4])
            for r in violation_rows
        ]

        combined = pass_samples + violation_samples
        random.shuffle(combined)
        print(f"[BalancedRandom] 最终样本: 正常={len(pass_samples)}, 违规={len(violation_samples)}, 总计={len(combined)}")
        return combined
    
    def load_balanced_samples(self, max_samples: int = 20000) -> List[Sample]:
        """
        加载样本并保持标签平衡（欠采样策略，不复制样本）
        
        策略：
        1. 获取两个标签的样本数量
        2. 取较小类别的数量作为每个标签的目标数量
        3. 从每个标签中随机抽取目标数量的样本
        4. 不复制任何样本，保证每个样本只出现一次
        5. 最终返回平衡的样本集（正常:违规 = 1:1）
        
        Args:
            max_samples: 最大样本数（必须是偶数，如果是奇数会向下取整）
            
        Returns:
            平衡的样本列表（不包含重复样本）
        """
        if max_samples <= 0:
            return []
        
        # 获取各标签的样本数量
        pass_count, violation_count = self.get_label_counts()
        
        # 如果某个标签完全没有样本，返回空列表
        if pass_count == 0 or violation_count == 0:
            print(f"[WARNING] 标签不平衡: 正常={pass_count}, 违规={violation_count}")
            print(f"[WARNING] 无法进行平衡采样，返回空列表")
            return []
        
        # 计算平衡数量（取较小类别的数量，实现欠采样）
        balanced_count = min(pass_count, violation_count)
        
        # 如果有 max_samples 限制，进一步限制每类的数量
        target_per_label = max_samples // 2
        if target_per_label > 0:
            balanced_count = min(balanced_count, target_per_label)
        
        if balanced_count == 0:
            print(f"[WARNING] 计算出的平衡数量为0，返回空列表")
            return []
        
        print(f"[BalancedSampling] 数据库样本分布: 正常={pass_count}, 违规={violation_count}")
        print(f"[BalancedSampling] 使用欠采样策略，每类抽取 {balanced_count} 个样本（不复制）")
        
        # 加载并随机抽取正常样本
        pass_samples = self._load_samples_by_label(0, pass_count)  # 先加载全部
        if len(pass_samples) > balanced_count:
            pass_samples = random.sample(pass_samples, balanced_count)  # 随机抽取
        print(f"[BalancedSampling] 正常样本: {len(pass_samples)} 个")
        
        # 加载并随机抽取违规样本
        violation_samples = self._load_samples_by_label(1, violation_count)  # 先加载全部
        if len(violation_samples) > balanced_count:
            violation_samples = random.sample(violation_samples, balanced_count)  # 随机抽取
        print(f"[BalancedSampling] 违规样本: {len(violation_samples)} 个")
        
        # 合并样本
        combined = pass_samples + violation_samples
        
        # 打乱顺序（重要：避免模型学到标签顺序）
        random.shuffle(combined)
        
        print(f"[BalancedSampling] 最终样本: 正常={len(pass_samples)}, 违规={len(violation_samples)}, 总计={len(combined)}")
        print(f"[BalancedSampling] ✓ 所有样本唯一，无重复")
        
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
        """根据文本查找样本（带重试机制）"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
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
            
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    print(f"[WARNING] 数据库锁定,重试 {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                raise
        
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
