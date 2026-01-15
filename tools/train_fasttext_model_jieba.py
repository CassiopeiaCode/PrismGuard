#!/usr/bin/env python3
"""
fastText 模型训练工具（jieba 分词版本）
用法: python tools/train_fasttext_model_jieba.py <profile_name>

注意：推荐使用 tools/train_fasttext_model.py，它会根据配置自动选择分词方式。

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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.fasttext_model_jieba import train_fasttext_model_jieba
from ai_proxy.moderation.smart.storage import SampleStorage


# 全局锁路径（所有 profile 和模型类型共用）
GLOBAL_LOCK_PATH = "configs/mod_profiles/.global_train.lock"


def _status_path(profile: ModerationProfile) -> str:
    return os.path.join(profile.base_dir, ".train_status.json")


def _log_path(profile: ModerationProfile) -> str:
    """训练日志路径"""
    return os.path.join(profile.base_dir, "train.log")


class FDTee:
    """
    文件描述符级别的 Tee，可以捕获 C 库的输出
    """
    def __init__(self, log_file, fd_num):
        """
        fd_num: 1 for stdout, 2 for stderr
        """
        self.log_file = log_file
        self.fd_num = fd_num
        self.original_fd = None
        self.pipe_read = None
        self.pipe_write = None
        self.reader_thread = None
        self.running = False
        self.last_cr_log_time = 0  # 上次记录含 \r 内容的时间
    
    def start(self):
        import threading
        
        # 保存原始文件描述符
        self.original_fd = os.dup(self.fd_num)
        
        # 创建管道
        self.pipe_read, self.pipe_write = os.pipe()
        
        # 将目标 fd 重定向到管道写端
        os.dup2(self.pipe_write, self.fd_num)
        
        # 启动读取线程
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
    
    def _reader_loop(self):
        """读取管道并同时写入原始 fd 和日志文件"""
        while self.running:
            try:
                data = os.read(self.pipe_read, 4096)
                if not data:
                    break
                # 写入原始输出（控制台）
                os.write(self.original_fd, data)
                # 写入日志文件（对含 \r 的高频输出限流）
                try:
                    text = data.decode('utf-8', errors='replace')
                    current_time = time.time()
                    
                    # 检查是否包含 \r（高频刷新的进度输出）
                    if '\r' in text and '\n' not in text:
                        # 含 \r 但不含 \n 的输出，每秒最多记录一次
                        if current_time - self.last_cr_log_time >= 1.0:
                            # 将 \r 替换为 \n 以便在日志中可读
                            self.log_file.write(text.replace('\r', '\n'))
                            self.log_file.flush()
                            self.last_cr_log_time = current_time
                    else:
                        # 正常输出，直接写入
                        self.log_file.write(text)
                        self.log_file.flush()
                except Exception:
                    pass
            except Exception:
                break
    
    def stop(self):
        self.running = False
        
        # 恢复原始文件描述符
        if self.original_fd is not None:
            os.dup2(self.original_fd, self.fd_num)
            os.close(self.original_fd)
        
        # 关闭管道
        if self.pipe_write is not None:
            os.close(self.pipe_write)
        if self.pipe_read is not None:
            os.close(self.pipe_read)
        
        # 等待读取线程结束
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1)


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
    
    # 确保全局锁目录存在
    os.makedirs(os.path.dirname(GLOBAL_LOCK_PATH), exist_ok=True)
    
    # 使用全局锁（所有 profile 和模型类型共用一个锁）
    try:
        lock_file = open(GLOBAL_LOCK_PATH, 'w')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={os.getpid()}\nprofile={profile_name}\nmodel=fasttext_jieba\ntime={int(time.time())}\n")
        lock_file.flush()
    except (IOError, OSError):
        print(f"[LOCK] 已有训练任务在进行中（全局锁），退出")
        if lock_file:
            lock_file.close()
        sys.exit(2)
    
    # 打开日志文件
    log_path = _log_path(profile)
    log_file = open(log_path, 'w', encoding='utf-8')
    
    # 使用文件描述符级别的 Tee 捕获 C 库输出
    stdout_tee = FDTee(log_file, 1)  # stdout
    stderr_tee = FDTee(log_file, 2)  # stderr
    stdout_tee.start()
    stderr_tee.start()
    
    try:
        start_time = datetime.now()
        print(f"{'='*50}")
        print(f"fastText 训练 (jieba): {profile_name}")
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
        
        _save_status(profile, 'started')
        
        train_fasttext_model_jieba(profile)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        _save_status(profile, 'completed')
        print(f"\n{'='*50}")
        print(f"✅ 训练完成: {profile.get_fasttext_model_path()}")
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
        # 停止文件描述符 Tee
        stdout_tee.stop()
        stderr_tee.stop()
        
        # 关闭日志文件
        log_file.close()
        
        # 释放全局锁
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


if __name__ == "__main__":
    main()
