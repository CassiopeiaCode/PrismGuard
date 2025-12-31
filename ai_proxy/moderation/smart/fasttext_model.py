"""
fastText 模型训练和预测模块
使用 fastText 进行文本分类,支持字符级和词级 n-gram

注意: 本模块使用新版 fastText API
- fasttext.load_model() 返回 FastText 对象（不再是 WordVectorModel 或 SupervisedModel）
- fasttext.train_supervised() 返回 FastText 对象
- 所有类型注解已更新为 fasttext.FastText
"""
import os
import tempfile
import fasttext
from typing import Dict, Tuple, Optional
from ai_proxy.moderation.smart.profile import ModerationProfile, SampleLoadingStrategy
from ai_proxy.moderation.smart.storage import SampleStorage


# 模型缓存：{profile_name: (model, model_mtime)}
_fasttext_cache: Dict[str, Tuple[fasttext.FastText, float]] = {}


def train_fasttext_model(profile: ModerationProfile):
    """
    训练 fastText 模型
    
    Args:
        profile: 配置对象
    """
    # 降低训练进程优先级，避免影响主服务
    try:
        original_nice = os.nice(0)
        os.nice(10)  # 提高nice值10，降低优先级
        print(f"[FastText] 训练进程优先级已调整 (nice: {original_nice} -> {os.nice(0)})")
    except Exception as e:
        print(f"[FastText] 无法调整进程优先级: {e}")
    
    storage = SampleStorage(profile.get_db_path())
    cfg = profile.config.fasttext_training
    
    # 数据库清理
    storage.cleanup_excess_samples(cfg.max_db_items)
    
    # 检查样本数量
    sample_count = storage.get_sample_count()
    if sample_count < cfg.min_samples:
        print(f"[FastText] 样本数不足 {cfg.min_samples}，当前={sample_count}，跳过训练")
        return
    
    # 加载样本（根据 profile 配置决定欠采样/全量）
    # 说明：
    # - balanced_undersample：欠采样随机平衡（每类<=max_samples/2）
    # - latest_full：平衡“全量模式”，每类取最新 max_samples/2
    # - random_full：平衡“全量随机模式”，每类随机抽 max_samples/2
    if cfg.sample_loading == SampleLoadingStrategy.latest_full:
        samples = storage.load_balanced_latest_samples(cfg.max_samples)
        print(f"[FastText] 样本加载策略: latest_full (balanced latest)")
    elif cfg.sample_loading == SampleLoadingStrategy.random_full:
        samples = storage.load_balanced_random_samples(cfg.max_samples)
        print(f"[FastText] 样本加载策略: random_full (balanced random)")
    else:
        samples = storage.load_balanced_samples(cfg.max_samples)
        print(f"[FastText] 样本加载策略: balanced_undersample")

    print(f"[FastText] 开始训练，共 {len(samples)} 个样本")
    
    # 准备训练数据文件
    train_file = _prepare_training_file(samples, profile)
    
    try:
        # 训练模型
        print(f"[FastText] 训练参数:")
        print(f"  维度: {cfg.dim}")
        print(f"  学习率: {cfg.lr}")
        print(f"  轮数: {cfg.epoch}")
        print(f"  词级 n-gram: {cfg.word_ngrams}")
        print(f"  字符级 n-gram: [{cfg.minn}, {cfg.maxn}]")
        
        model = fasttext.train_supervised(
            input=train_file,
            dim=cfg.dim,
            lr=cfg.lr,
            epoch=cfg.epoch,
            wordNgrams=cfg.word_ngrams,
            minn=cfg.minn,
            maxn=cfg.maxn,
            bucket=cfg.bucket,
            loss="ova",  # one-vs-all，适合二分类
            thread=2,  # 限制为单线程，降低 CPU 使用率
            verbose=2
        )
        
        # 保存模型（使用临时文件 + 原子替换，避免产生不完整的模型文件）
        model_path = profile.get_fasttext_model_path()
        temp_model_path = model_path + ".tmp"
        
        # 先保存到临时文件
        model.save_model(temp_model_path)
        
        # 验证临时文件
        if not os.path.exists(temp_model_path):
            raise RuntimeError("模型保存失败：临时文件不存在")
        
        temp_size = os.path.getsize(temp_model_path)
        if temp_size < 1024:
            os.remove(temp_model_path)
            raise RuntimeError(f"模型保存失败：文件过小 ({temp_size} bytes)")
        
        # 尝试加载临时文件验证完整性
        try:
            test_model = fasttext.load_model(temp_model_path)
            test_labels, _ = test_model.predict("验证测试", k=1)
            if not test_labels:
                raise RuntimeError("模型验证失败：预测返回空结果")
            del test_model
        except Exception as e:
            os.remove(temp_model_path)
            raise RuntimeError(f"模型验证失败：{e}")
        
        # 原子替换（Windows 上需要先删除目标文件）
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(temp_model_path, model_path)
        
        print(f"[FastText] 模型已保存: {model_path} ({temp_size / 1024:.1f} KB)")
        
        # 评估
        result = model.test(train_file)
        print(f"[FastText] 训练集评估:")
        print(f"  样本数: {result[0]}")
        print(f"  准确率: {result[1]:.3f}")
        print(f"  召回率: {result[2]:.3f}")
        
    finally:
        # 清理临时文件
        if os.path.exists(train_file):
            os.remove(train_file)


def _prepare_training_file(samples, profile: ModerationProfile) -> str:
    """
    准备 fastText 训练文件
    
    格式: __label__0 文本内容
          __label__1 文本内容
    
    Returns:
        训练文件路径
    """
    # 创建临时文件
    fd, train_file = tempfile.mkstemp(suffix=".txt", prefix="fasttext_train_")
    
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for sample in samples:
            # fastText 格式: __label__<类别> <文本>
            # 文本需要预处理: 去除换行符
            text = sample.text.replace('\n', ' ').replace('\r', ' ')
            label = sample.label  # 0 或 1
            f.write(f"__label__{label} {text}\n")
    
    print(f"[FastText] 训练文件已生成: {train_file}")
    return train_file


def _load_fasttext_with_cache(profile: ModerationProfile) -> fasttext.FastText:
    """
    加载 fastText 模型（带缓存）
    
    注意: fasttext.load_model() 现在返回 FastText 对象，
    而不是旧版本的 WordVectorModel 或 SupervisedModel
    
    改进：
    1. 添加模型文件完整性检查
    2. 加载失败时清理缓存并抛出明确异常
    3. 验证模型能够正常预测
    4. 发现损坏模型时自动删除，以便下次调度器重新训练
    
    Returns:
        fastText 模型对象 (FastText)
        
    Raises:
        FileNotFoundError: 模型文件不存在
        RuntimeError: 模型文件损坏或无法加载
    """
    profile_name = profile.profile_name
    model_path = profile.get_fasttext_model_path()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"fastText 模型不存在: {model_path}")
    
    # 检查文件大小（避免加载空文件或损坏文件）
    file_size = os.path.getsize(model_path)
    if file_size < 1024:  # 小于 1KB 认为是无效文件
        print(f"[ERROR] fastText 模型文件过小，删除损坏文件: {model_path}")
        _remove_corrupted_model(model_path)
        raise RuntimeError(f"fastText 模型文件过小或损坏 ({file_size} bytes): {model_path}")
    
    # 获取文件修改时间
    model_mtime = os.path.getmtime(model_path)
    
    # 检查缓存
    if profile_name in _fasttext_cache:
        cached_model, cached_mtime = _fasttext_cache[profile_name]
        
        # 如果文件没有更新，重用缓存
        if model_mtime == cached_mtime:
            print(f"[DEBUG] 重用缓存的 fastText 模型: {profile_name}")
            return cached_model
        else:
            print(f"[DEBUG] fastText 模型文件已更新，重新加载: {profile_name}")
            del _fasttext_cache[profile_name]
    
    # 加载模型 (返回 FastText 对象，兼容新版 API)
    print(f"[DEBUG] 加载 fastText 模型: {model_path}")
    try:
        model = fasttext.load_model(model_path)
    except Exception as e:
        # 清理可能的损坏缓存
        if profile_name in _fasttext_cache:
            del _fasttext_cache[profile_name]
        print(f"[ERROR] fastText 模型加载失败，删除损坏文件: {model_path}")
        _remove_corrupted_model(model_path)
        raise RuntimeError(f"fastText 模型加载失败: {model_path}, 错误: {e}")
    
    # 验证模型能够正常工作
    try:
        labels, probs = model.predict("验证测试", k=1)
        if not labels:
            raise RuntimeError("模型预测返回空结果")
    except Exception as e:
        print(f"[ERROR] fastText 模型验证失败，删除损坏文件: {model_path}")
        _remove_corrupted_model(model_path)
        raise RuntimeError(f"fastText 模型验证失败: {model_path}, 错误: {e}")
    
    # 保存到缓存
    _fasttext_cache[profile_name] = (model, model_mtime)
    
    return model


def fasttext_predict_proba(text: str, profile: ModerationProfile) -> float:
    """
    使用 fastText 模型预测违规概率
    
    Args:
        text: 待预测文本
        profile: 配置对象
        
    Returns:
        违规概率 (0-1)
    """
    print(f"[DEBUG] fastText 模型预测")
    
    # 加载模型（带缓存）
    model = _load_fasttext_with_cache(profile)
    
    # 预处理文本
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 预测（返回 top-k 标签和概率）
    labels, probs = model.predict(text, k=2)
    
    print(f"  预测标签: {labels}")
    print(f"  预测概率: {probs}")
    
    # 找出"违规"标签（__label__1）的概率
    violation_prob = None
    safe_prob = None
    
    for label, p in zip(labels, probs):
        if label == "__label__1":
            violation_prob = float(p)
        elif label == "__label__0":
            safe_prob = float(p)
    
    # 如果没有直接返回违规概率，但有正常概率，则计算违规概率
    if violation_prob is None:
        if safe_prob is not None:
            # 违规概率 = 1 - 正常概率
            violation_prob = 1.0 - safe_prob
        else:
            # 边缘情况：两个标签都没有返回（不应该发生，但作为后备）
            violation_prob = 0.0
    
    print(f"  违规概率: {violation_prob:.3f}")
    return violation_prob


def fasttext_model_exists(profile: ModerationProfile) -> bool:
    """检查 fastText 模型是否存在"""
    return os.path.exists(profile.get_fasttext_model_path())


def _remove_corrupted_model(model_path: str) -> None:
    """删除损坏的模型文件，以便调度器下次重新训练"""
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"[INFO] 已删除损坏的模型文件: {model_path}")
    except Exception as e:
        print(f"[WARNING] 无法删除损坏的模型文件: {model_path}, 错误: {e}")