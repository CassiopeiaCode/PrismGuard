#!/usr/bin/env python3
"""
fastText 模型训练工具
用法: python tools/train_fasttext_model.py <profile_name>

Exit codes:
  0: 训练完成
  1: 训练失败/异常
  2: 锁占用/已有训练进行中

注意：全局只允许同时运行一个训练任务（跨所有 profile）
"""
import sys
import os
import time
import json
import fcntl
import io
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.fasttext_model import train_fasttext_model
from ai_proxy.moderation.smart.fasttext_model_jieba import train_fasttext_model_jieba
from ai_proxy.moderation.smart.storage import SampleStorage


# 全局锁路径（所有 profile 共用）
GLOBAL_LOCK_PATH = "configs/mod_profiles/.global_train.lock"


def _status_path(profile: ModerationProfile) -> str:
    return os.path.join(profile.base_dir, ".train_status.json")


def _log_path(profile: ModerationProfile) -> str:
    """训练日志路径"""
    return os.path.join(profile.base_dir, "train.log")


def _save_status(profile: ModerationProfile, status: str, error: str = None):
    """保存训练状态"""
    data = {
        'status': status,
        'timestamp': int(time.time()),
        'pid': os.getpid(),
        'model_type': 'fasttext',
    }
    if error:
        data['error'] = str(error)[:500]
    try:
        with open(_status_path(profile), 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def _validate_model(model_path: str) -> bool:
    """验证模型文件"""
    import fasttext
    if not os.path.exists(model_path):
        return False
    if os.path.getsize(model_path) < 1024:
        return False
    try:
        model = fasttext.load_model(model_path)
        labels, _ = model.predict("test", k=1)
        return bool(labels)
    except Exception:
        return False


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


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/train_fasttext_model.py <profile_name>")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    profile = ModerationProfile(profile_name)
    lock_file = None
    
    # 确保全局锁目录存在
    os.makedirs(os.path.dirname(GLOBAL_LOCK_PATH), exist_ok=True)
    
    # 使用全局锁（所有 profile 共用一个锁）
    try:
        lock_file = open(GLOBAL_LOCK_PATH, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={os.getpid()}\nprofile={profile_name}\ntime={int(time.time())}\n")
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
        
        start_time = datetime.now()
        print(f"{'='*50}")
        print(f"fastText 训练: {profile_name}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        cfg = profile.config.fasttext_training
        storage = SampleStorage(profile.get_db_path())
        sample_count = storage.get_sample_count()
        
        print(f"样本数: {sample_count}, 最小要求: {cfg.min_samples}")
        
        if sample_count < cfg.min_samples:
            print(f"样本不足，跳过")
            lock_file.close()
            log_file.close()
            sys.exit(1)
        
        # 选择训练函数
        if cfg.use_jieba or cfg.use_tiktoken:
            train_func = train_fasttext_model_jieba
            print(f"分词模式: {'jieba' if cfg.use_jieba else ''}{'+' if cfg.use_jieba and cfg.use_tiktoken else ''}{'tiktoken' if cfg.use_tiktoken else ''}")
        else:
            train_func = train_fasttext_model
            print(f"分词模式: 字符级 n-gram")
        
        _save_status(profile, 'started')
        
        train_func(profile)
        
        model_path = profile.get_fasttext_model_path()
        if not _validate_model(model_path):
            raise RuntimeError("模型验证失败")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        _save_status(profile, 'completed')
        print(f"\n{'='*50}")
        print(f"✅ 训练完成: {model_path}")
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
