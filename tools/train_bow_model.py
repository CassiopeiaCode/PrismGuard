#!/usr/bin/env python3
"""
词袋模型训练工具
用法: python tools/train_bow_model.py <profile_name>

Exit codes:
  0: 训练完成
  1: 训练失败/异常
  2: 锁占用/已有训练进行中

注意：全局只允许同时运行一个训练任务（跨所有 profile 和模型类型）
"""
import sys
import os
import time
import json
import fcntl

sys.path.insert(0, ".")

from ai_proxy.moderation.smart.profile import get_profile, ModerationProfile
from ai_proxy.moderation.smart.bow import train_bow_model


# 全局锁路径（所有 profile 和模型类型共用）
GLOBAL_LOCK_PATH = "configs/mod_profiles/.global_train.lock"


def _status_path(profile: ModerationProfile) -> str:
    return os.path.join(profile.base_dir, ".train_status.json")


def _log_path(profile: ModerationProfile) -> str:
    """训练日志路径"""
    return os.path.join(profile.base_dir, "train.log")


class TeeWriter:
    """同时写入多个输出流"""
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, text):
        for w in self.writers:
            w.write(text)
            w.flush()
    
    def flush(self):
        for w in self.writers:
            w.flush()


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
    
    # 确保全局锁目录存在
    os.makedirs(os.path.dirname(GLOBAL_LOCK_PATH), exist_ok=True)
    
    # 使用全局锁（所有 profile 和模型类型共用一个锁）
    try:
        lock_file = open(GLOBAL_LOCK_PATH, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={os.getpid()}\nprofile={profile_name}\nmodel=bow\ntime={int(time.time())}\n")
        lock_file.flush()
    except (IOError, OSError):
        print(f"[LOCK] 已有训练任务在进行中（全局锁），退出")
        if lock_file:
            lock_file.close()
        sys.exit(2)
    
    # 打开日志文件
    log_path = _log_path(profile)
    log_file = open(log_path, 'w', encoding='utf-8')
    
    # 创建 Tee 写入器，同时输出到控制台和日志文件
    tee_out = TeeWriter(sys.stdout, log_file)
    tee_err = TeeWriter(sys.stderr, log_file)
    
    # 保存原始 stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # 重定向输出
        sys.stdout = tee_out
        sys.stderr = tee_err
        
        from datetime import datetime
        start_time = datetime.now()
        print(f"{'='*50}")
        print(f"BoW 训练: {profile_name}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        _save_status(profile, 'started')
        
        train_bow_model(profile)
        
        if not _validate_model(profile):
            raise RuntimeError("模型验证失败")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        _save_status(profile, 'completed')
        print(f"\n{'='*50}")
        print(f"✅ 训练完成")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"耗时: {duration:.1f} 秒")
        print(f"{'='*50}")
        
    except Exception as e:
        _save_status(profile, 'failed', str(e))
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 恢复原始 stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # 关闭日志文件
        log_file.close()
        
        # 释放全局锁
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


if __name__ == "__main__":
    main()
