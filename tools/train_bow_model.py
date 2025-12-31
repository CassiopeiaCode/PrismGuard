#!/usr/bin/env python3
"""
词袋模型训练工具
用法: python tools/train_bow_model.py <profile_name>

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

sys.path.insert(0, ".")

from ai_proxy.moderation.smart.profile import get_profile, ModerationProfile
from ai_proxy.moderation.smart.bow import train_bow_model


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


def _validate_model(profile: ModerationProfile) -> bool:
    """验证模型文件"""
    import joblib
    model_path = profile.get_model_path()
    vec_path = profile.get_vectorizer_path()
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return False
    if os.path.getsize(model_path) < 100 or os.path.getsize(vec_path) < 100:
        return False
    try:
        vec = joblib.load(vec_path)
        clf = joblib.load(model_path)
        X = vec.transform(["test"])
        clf.predict_proba(X)
        return True
    except Exception:
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/train_bow_model.py <profile_name>")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    profile = get_profile(profile_name)
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
    print(f"BoW 训练: {profile_name}")
    print(f"{'='*50}")
    
    _save_status(profile, 'started')
    
    try:
        train_bow_model(profile)
        
        if not _validate_model(profile):
            raise RuntimeError("模型验证失败")
        
        _save_status(profile, 'completed')
        print(f"\n✅ 训练完成")
        
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
