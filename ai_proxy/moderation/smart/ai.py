"""
AI 审核模块 - 支持三段式决策 (修复版)
"""
import os
import json
import random
import hashlib
import threading
from collections import OrderedDict
from typing import Tuple, Optional, Dict
from pydantic import BaseModel
from openai import AsyncOpenAI

from ai_proxy.config import settings
from ai_proxy.moderation.smart.profile import get_profile, ModerationProfile
from ai_proxy.moderation.smart.storage import SampleStorage
from ai_proxy.utils.memory_guard import track_container, check_container


def local_model_predict_proba(text: str, profile: ModerationProfile) -> float:
    """
    本地模型预测接口（统一抽象）
    
    根据配置的 local_model_type 决定使用哪种模型
    
    Args:
        text: 待预测文本
        profile: 配置对象
        
    Returns:
        违规概率 (0-1)
    """
    from ai_proxy.moderation.smart.profile import LocalModelType
    
    model_type = profile.config.local_model_type
    
    if model_type == LocalModelType.fasttext:
        from ai_proxy.moderation.smart.fasttext_model import fasttext_predict_proba
        return fasttext_predict_proba(text, profile)
    else:  # 默认使用 BoW
        from ai_proxy.moderation.smart.bow import bow_predict_proba
        return bow_predict_proba(text, profile)


class ModerationResult(BaseModel):
    """审核结果"""
    violation: bool
    category: Optional[str] = None
    reason: Optional[str] = None
    source: str  # "ai" or "bow_model"
    confidence: Optional[float] = None


# LRU 缓存池：每个配置一个缓存
_moderation_cache: Dict[str, OrderedDict] = {}
_cache_lock = threading.Lock()
CACHE_SIZE = 20
MAX_PROFILES = 50  # ✅ 限制最大配置数量


# 全局 OpenAI 客户端池
_openai_clients: Dict[str, AsyncOpenAI] = {}
_client_lock = threading.Lock()


def get_or_create_openai_client(profile: ModerationProfile) -> AsyncOpenAI:
    """获取或创建 OpenAI 客户端（复用连接）"""
    api_key = getattr(settings, profile.config.ai.api_key_env, None)
    if not api_key:
        raise ValueError(f"Environment variable {profile.config.ai.api_key_env} not set")
    
    # 使用 base_url 和 api_key 作为缓存键
    cache_key = f"{profile.config.ai.base_url}:{api_key[:10]}"
    
    with _client_lock:
        if cache_key not in _openai_clients:
            _openai_clients[cache_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=profile.config.ai.base_url,
                timeout=profile.config.ai.timeout
            )
    
    return _openai_clients[cache_key]


async def cleanup_openai_clients():
    """清理所有 OpenAI 客户端（应用关闭时调用）"""
    with _client_lock:
        for client in _openai_clients.values():
            await client.close()
        _openai_clients.clear()


def _get_text_hash(text: str) -> str:
    """计算文本哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _get_cache(profile_name: str) -> OrderedDict:
    """获取或创建指定配置的缓存"""
    with _cache_lock:
        if profile_name not in _moderation_cache:
            # ✅ 限制缓存字典大小
            if len(_moderation_cache) >= MAX_PROFILES:
                # 删除最老的配置缓存（FIFO）
                oldest = next(iter(_moderation_cache))
                _moderation_cache.pop(oldest)
                print(f"[DEBUG] 缓存字典已满，移除配置: {oldest}")
            
            _moderation_cache[profile_name] = OrderedDict()
            # 追踪新创建的缓存
            track_container(_moderation_cache[profile_name], f"moderation_cache.{profile_name}")
        
        # 定期检查整个缓存字典
        check_container(_moderation_cache, "moderation_cache_dict")
        
        return _moderation_cache[profile_name]


def _check_cache(profile_name: str, text: str) -> Optional[ModerationResult]:
    """检查缓存中是否存在审核结果"""
    cache = _get_cache(profile_name)
    text_hash = _get_text_hash(text)
    
    with _cache_lock:
        if text_hash in cache:
            # 命中缓存，移到末尾（最近使用）
            cache.move_to_end(text_hash)
            result = cache[text_hash]
            print(f"[DEBUG] 缓存命中: hash={text_hash[:8]}... 结果={'✅通过' if not result.violation else '❌违规'}")
            return result
    
    return None


def _save_to_cache(profile_name: str, text: str, result: ModerationResult):
    """保存审核结果到缓存"""
    cache = _get_cache(profile_name)
    text_hash = _get_text_hash(text)
    
    with _cache_lock:
        # 添加到缓存
        cache[text_hash] = result
        cache.move_to_end(text_hash)
        
        # LRU 淘汰：如果超过大小限制，删除最老的
        if len(cache) > CACHE_SIZE:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key)
            print(f"[DEBUG] 缓存满，淘汰最旧项: hash={oldest_key[:8]}...")
        
        print(f"[DEBUG] 保存到缓存: hash={text_hash[:8]}... 当前缓存大小={len(cache)}/{CACHE_SIZE}")


async def ai_moderate(text: str, profile: ModerationProfile) -> ModerationResult:
    """使用 AI 进行审核（复用客户端）"""
    print(f"[MODERATION] AI审核开始")
    print(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    client = get_or_create_openai_client(profile)  # ✅ 复用客户端
    prompt = profile.render_prompt(text)
    
    try:
        response = await client.chat.completions.create(
            model=profile.config.ai.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        
        # 智能提取 JSON：删除第一个 { 前和最后一个 } 后的内容
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                violation = any(word in content.lower() for word in ["违规", "violation", "不当"])
                data = {"violation": violation, "category": "unknown", "reason": content[:200]}
        else:
            violation = any(word in content.lower() for word in ["违规", "violation", "不当"])
            data = {"violation": violation, "category": "unknown", "reason": content[:200]}
        
        result = ModerationResult(
            violation=data.get("violation", False),
            category=data.get("category"),
            reason=data.get("reason"),
            source="ai"
        )
        print(f"[MODERATION] AI审核结果: {'❌ 违规' if result.violation else '✅ 通过'}")
        if result.category:
            print(f"  类别: {result.category}")
        if result.reason:
            print(f"  原因: {result.reason[:100]}{'...' if len(result.reason) > 100 else ''}")
        return result
    
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[ERROR] AI moderation exception:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise


async def run_ai_moderation_and_log(text: str, profile: ModerationProfile) -> ModerationResult:
    """AI 审核并记录结果（先查数据库，避免重复调用AI）"""
    storage = SampleStorage(profile.get_db_path())
    
    # 先查数据库，如果已有记录则直接返回
    existing_sample = storage.find_by_text(text)
    if existing_sample:
        print(f"[DEBUG] 数据库命中: 文本已审核过")
        result = ModerationResult(
            violation=bool(existing_sample.label),
            category=existing_sample.category,
            reason=f"From DB: {existing_sample.created_at}",
            source="ai",
            confidence=None
        )
        print(f"[MODERATION] 数据库结果: {'❌ 违规' if result.violation else '✅ 通过'}")
        return result
    
    # 数据库没有，调用AI审核
    print(f"[DEBUG] 数据库未命中，调用AI审核")
    result = await ai_moderate(text, profile)
    
    # 保存到数据库
    label = 1 if result.violation else 0
    storage.save_sample(text, label, result.category)
    print(f"[DEBUG] 审核结果已保存到数据库")
    
    return result


async def smart_moderation(text: str, cfg: dict) -> Tuple[bool, Optional[ModerationResult]]:
    """
    智能审核入口 - 三段式决策 + LRU缓存
    
    流程：
    0. 检查缓存，命中则直接返回
    1. 随机抽样 -> AI 审核并记录
    2. 本地词袋模型 -> 低风险放行 / 高风险拒绝 / 中间交 AI
    3. 无模型 -> 全部交 AI
    4. 保存结果到缓存
    """
    if not cfg.get("enabled", False):
        print(f"[DEBUG] 智能审核: 未启用，跳过")
        return True, None
    
    print(f"[DEBUG] 智能审核开始")
    
    profile_name = cfg.get("profile", "default")
    profile = get_profile(profile_name)
    
    # 统一截断文本
    original_len = len(text)
    text = profile.truncate_text(text)
    if len(text) < original_len:
        print(f"  文本已截断: {original_len} -> {len(text)}")
    
    print(f"  待审核文本: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # 0. 检查缓存
    cached_result = _check_cache(profile_name, text)
    if cached_result is not None:
        return not cached_result.violation, cached_result
    
    print(f"  使用配置: {profile_name}")
    print(f"  AI审核概率: {profile.config.probability.ai_review_rate * 100:.1f}%")
    
    ai_rate = profile.config.probability.ai_review_rate
    rand_val = random.random()
    
    # 1. 随机抽样：直接走 AI（用于持续产生标注）
    if rand_val < ai_rate:
        print(f"[DEBUG] 决策路径: 随机抽样 (rand={rand_val:.3f} < {ai_rate:.3f}) -> AI审核")
        result = await run_ai_moderation_and_log(text, profile)
        _save_to_cache(profile_name, text, result)
        return not result.violation, result
    
    # 2. 尝试本地模型
    if profile.local_model_exists():
        model_type = profile.config.local_model_type
        print(f"[DEBUG] 决策路径: 本地模型预测 (类型={model_type.value})")
        
        try:
            p = local_model_predict_proba(text, profile)
            low_t = profile.config.probability.low_risk_threshold
            high_t = profile.config.probability.high_risk_threshold
            
            print(f"  违规概率: {p:.3f}")
            print(f"  阈值: 低风险 < {low_t:.3f}, 高风险 > {high_t:.3f}")
            
            # 低风险：直接放行
            if p < low_t:
                print(f"[DEBUG] 本地模型结果: ✅ 低风险放行")
                result = ModerationResult(
                    violation=False,
                    reason=f"{model_type.value}: low risk (p={p:.3f})",
                    source=f"{model_type.value}_model",
                    confidence=p
                )
                _save_to_cache(profile_name, text, result)
                return True, result
            
            # 高风险：直接拒绝
            if p > high_t:
                print(f"[DEBUG] 本地模型结果: ❌ 高风险拒绝")
                result = ModerationResult(
                    violation=True,
                    reason=f"{model_type.value}: high risk (p={p:.3f})",
                    source=f"{model_type.value}_model",
                    confidence=p
                )
                _save_to_cache(profile_name, text, result)
                return False, result
            
            # 不确定：交给 AI 复核
            print(f"[DEBUG] 本地模型结果: ⚠️ 不确定 -> AI复核")
            result = await run_ai_moderation_and_log(text, profile)
            _save_to_cache(profile_name, text, result)
            return not result.violation, result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[DEBUG] 本地模型预测失败: {error_msg}")
            
            # 检查是否是 NumPy 2.0 兼容性问题
            if "copy" in error_msg and "np.array" in error_msg:
                print(f"[WARNING] 检测到 NumPy 2.0 兼容性问题!")
                print(f"[WARNING] fastText 库尚未完全兼容 NumPy 2.0")
                print(f"[WARNING] 建议降级 NumPy: pip install 'numpy<2.0'")
                print(f"[WARNING] 或等待 fastText 更新以支持 NumPy 2.0")
            
            # 打印详细堆栈信息（仅在首次出现时）
            if not hasattr(local_model_predict_proba, '_error_logged'):
                print(f"[DEBUG] 详细错误信息:")
                traceback.print_exc()
                local_model_predict_proba._error_logged = True
            
            print(f"[DEBUG] 回退到 AI 审核")
    else:
        print(f"[DEBUG] 决策路径: 无本地模型 -> AI审核")
    
    # 3. 无模型或失败：全部交 AI
    result = await run_ai_moderation_and_log(text, profile)
    _save_to_cache(profile_name, text, result)
    return not result.violation, result