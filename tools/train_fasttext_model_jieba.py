#!/usr/bin/env python3
"""
fastText 模型训练工具（jieba 分词版本）
用法: python tools/train_fasttext_model_jieba.py <profile_name>

注意：推荐使用 tools/train_fasttext_model.py，它会根据配置自动选择分词方式。

Exit codes:
  0: 训练完成
  1: 训练失败/异常
  2: 锁占用/已有训练进行中
"""
import sys
import os
import time
import json
import fcntl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.fasttext_model_jieba import train_fasttext_model_jieba
from ai_proxy.moderation.smart.storage import SampleStorage


def _lock_path(profile: ModerationProfile) -> str:
    return os.path.join(profile.base_dir, ".train.lock")


def _status_path(profile: ModerationProfile) -> str:
    return os.path.join(profile.base_dir, ".train_status.json")


def _save_status(profile: ModerationProfile, status: str, error: str = None):
    """保存训练状态"""
    data = {
        'status': status,
        'timestamp': int(time.time()),
        'pid': os.getpid(),
    }
    if error:
        data['error'] = str(error)[:500]
    try:
        with open(_status_path(profile), 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/train_fasttext_model_jieba.py <profile_name>")
        print("注意：推荐使用 tools/train_fasttext_model.py")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    profile = ModerationProfile(profile_name)
    lock_file = None
    
    # 使用 fcntl.flock 获取独占锁
    lock_path = _lock_path(profile)
    try:
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={os.getpid()}\ntime={int(time.time())}\n")
        lock_file.flush()
    except (IOError, OSError):
        print(f"[LOCK] 已有训练在进行中，退出")
        if lock_file:
            lock_file.close()
        sys.exit(2)
    
    print(f"{'='*50}")
    print(f"fastText 训练 (jieba): {profile_name}")
    print(f"{'='*50}")
    
    cfg = profile.config.fasttext_training
    storage = SampleStorage(profile.get_db_path())
    sample_count = storage.get_sample_count()
    
    print(f"样本数: {sample_count}, 最小要求: {cfg.min_samples}")
    
    if sample_count < cfg.min_samples:
        print(f"样本不足，跳过")
        lock_file.close()
        sys.exit(1)
    
    _save_status(profile, 'started')
    
    try:
        train_fasttext_model_jieba(profile)
        _save_status(profile, 'completed')
        print(f"\n✅ 训练完成: {profile.get_fasttext_model_path()}")
        
    except Exception as e:
        _save_status(profile, 'failed', str(e))
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


if __name__ == "__main__":
    main()
