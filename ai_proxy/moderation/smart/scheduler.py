"""
定时任务调度器 - 自动训练词袋模型
"""
import os
import asyncio
import time
from datetime import datetime
from typing import List, Dict

from ai_proxy.moderation.smart.profile import ModerationProfile, LocalModelType
from ai_proxy.moderation.smart.bow import train_bow_model
from ai_proxy.moderation.smart.storage import SampleStorage

# 记录每个 profile 的训练锁，避免重复训练
_profile_locks: Dict[str, asyncio.Lock] = {}


def get_profile_lock(profile_name: str) -> asyncio.Lock:
    """获取对应 profile 的训练锁"""
    if profile_name not in _profile_locks:
        _profile_locks[profile_name] = asyncio.Lock()
    return _profile_locks[profile_name]


def get_all_profiles() -> List[str]:
    """扫描所有配置文件夹"""
    base_dir = "configs/mod_profiles"
    if not os.path.exists(base_dir):
        return []
    
    profiles = []
    for name in os.listdir(base_dir):
        profile_dir = os.path.join(base_dir, name)
        if os.path.isdir(profile_dir):
            config_file = os.path.join(profile_dir, "profile.json")
            if os.path.exists(config_file):
                profiles.append(name)
    
    return profiles


def train_local_model(profile: ModerationProfile):
    """
    训练本地模型（根据配置类型）
    
    Args:
        profile: 配置对象
    """
    model_type = profile.config.local_model_type
    
    if model_type == LocalModelType.fasttext:
        from ai_proxy.moderation.smart.fasttext_model import train_fasttext_model
        train_fasttext_model(profile)
    else:  # 默认 BoW
        train_bow_model(profile)


def should_train(profile: ModerationProfile) -> bool:
    """判断是否需要训练"""
    model_type = profile.config.local_model_type
    
    # 获取对应模型的配置
    if model_type == LocalModelType.fasttext:
        cfg = profile.config.fasttext_training
        model_path = profile.get_fasttext_model_path()
    else:
        cfg = profile.config.bow_training
        model_path = profile.get_model_path()
    
    # 检查样本数量
    storage = SampleStorage(profile.get_db_path())
    sample_count = storage.get_sample_count()
    
    if sample_count < cfg.min_samples:
        return False
    
    # 检查模型是否存在
    if not profile.local_model_exists():
        return True
    
    # 检查模型文件修改时间
    model_mtime = os.path.getmtime(model_path)
    interval_seconds = cfg.retrain_interval_minutes * 60
    
    return (time.time() - model_mtime) > interval_seconds


async def train_all_profiles():
    """训练所有需要训练的配置"""
    profiles = get_all_profiles()
    
    if not profiles:
        print(f"[SCHEDULER] 未找到配置文件")
        return
    
    print(f"[SCHEDULER] 扫描到 {len(profiles)} 个配置: {', '.join(profiles)}")
    
    for profile_name in profiles:
        try:
            profile = ModerationProfile(profile_name)
            
            if should_train(profile):
                lock = get_profile_lock(profile_name)
                
                if lock.locked():
                    print(f"[SCHEDULER] {profile_name} 正在训练中，跳过本次调度")
                    continue
                
                model_type = profile.config.local_model_type
                print(f"[SCHEDULER] 开始训练: {profile_name} (模型类型={model_type.value})")
                async with lock:
                    await asyncio.to_thread(train_local_model, profile)
                print(f"[SCHEDULER] 训练完成: {profile_name}")
            else:
                storage = SampleStorage(profile.get_db_path())
                sample_count = storage.get_sample_count()
                print(f"[SCHEDULER] 跳过训练: {profile_name} (样本数={sample_count})")
        
        except Exception as e:
            print(f"[SCHEDULER] 训练失败: {profile_name} - {e}")


async def scheduler_loop(check_interval_minutes: int = 10):
    """定时任务循环"""
    print(f"[SCHEDULER] 启动定时任务，检查间隔: {check_interval_minutes} 分钟")
    
    while True:
        try:
            print(f"[SCHEDULER] 开始检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await train_all_profiles()
        except Exception as e:
            print(f"[SCHEDULER] 任务执行失败: {e}")
        
        await asyncio.sleep(check_interval_minutes * 60)


def start_scheduler(check_interval_minutes: int = 10):
    """启动调度器（在后台任务中运行）"""
    asyncio.create_task(scheduler_loop(check_interval_minutes))
