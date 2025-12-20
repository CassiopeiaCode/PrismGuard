"""
fastText 模型训练和预测模块（高级分词版本）
支持多种分词方式：jieba、tiktoken、组合分词

分词模式：
1. jieba only: 使用 jieba 中文分词
2. tiktoken only: 使用 tiktoken BPE 分词
3. tiktoken + jieba: 先 tiktoken 再 jieba（实验性）

相比原版的改进：
- 支持多种分词方式
- 关闭子词 n-gram (minn=0, maxn=0)
- 使用词级 n-gram 提取特征
- 使用 tqdm 显示分词进度
"""
import os
import tempfile
import fasttext
import jieba
import tiktoken
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from ai_proxy.moderation.smart.profile import ModerationProfile, SampleLoadingStrategy
from ai_proxy.moderation.smart.storage import SampleStorage


# tiktoken 编码器缓存
_tiktoken_encoders: Dict[str, tiktoken.Encoding] = {}


def get_tiktoken_encoder(model_name: str = "cl100k_base") -> tiktoken.Encoding:
    """获取或创建 tiktoken 编码器（带缓存）"""
    if model_name not in _tiktoken_encoders:
        _tiktoken_encoders[model_name] = tiktoken.get_encoding(model_name)
    return _tiktoken_encoders[model_name]


def tokenize_text(text: str, use_jieba: bool, use_tiktoken: bool, tiktoken_model: str = "cl100k_base") -> str:
    """
    根据配置对文本进行分词
    
    Args:
        text: 待分词文本
        use_jieba: 是否使用 jieba
        use_tiktoken: 是否使用 tiktoken
        tiktoken_model: tiktoken 模型名称
        
    Returns:
        分词后的文本（空格分隔）
    """
    if use_tiktoken and use_jieba:
        # 组合模式：先 tiktoken 再 jieba
        encoder = get_tiktoken_encoder(tiktoken_model)
        tokens = encoder.encode(text)
        # 将 token ID 转回文本片段
        token_texts = [encoder.decode([t]) for t in tokens]
        # 对每个片段再用 jieba 分词
        final_words = []
        for token_text in token_texts:
            words = list(jieba.cut(token_text))
            final_words.extend(words)
        return ' '.join(final_words)
    
    elif use_tiktoken:
        # 仅 tiktoken
        encoder = get_tiktoken_encoder(tiktoken_model)
        # 训练数据里可能包含类似 "<|endoftext|>" 的文本片段；tiktoken 默认会把它视为“特殊 token”并报错
        # 这里禁用 disallowed_special 检查，把它当作普通文本编码，避免训练中断
        tokens = encoder.encode(text, disallowed_special=())
        # 将 token ID 转为字符串（保留 ID 作为特征）
        return ' '.join([f"tk{t}" for t in tokens])
    
    elif use_jieba:
        # 仅 jieba
        words = jieba.cut(text)
        return ' '.join(words)
    
    else:
        # 不应该到这里，但作为后备返回原文
        return text


# 模型缓存：{profile_name: (model, model_mtime)}
_fasttext_cache: Dict[str, Tuple[fasttext.FastText, float]] = {}


def train_fasttext_model_jieba(profile: ModerationProfile):
    """
    训练 fastText 模型（jieba 分词版本）
    
    Args:
        profile: 配置对象
    """
    # 降低训练进程优先级，避免影响主服务
    try:
        original_nice = os.nice(0)
        os.nice(10)
        print(f"[FastText-Jieba] 训练进程优先级已调整 (nice: {original_nice} -> {os.nice(0)})")
    except Exception as e:
        print(f"[FastText-Jieba] 无法调整进程优先级: {e}")
    
    storage = SampleStorage(profile.get_db_path())
    cfg = profile.config.fasttext_training
    
    # 数据库清理
    storage.cleanup_excess_samples(cfg.max_db_items)
    
    # 检查样本数量
    sample_count = storage.get_sample_count()
    if sample_count < cfg.min_samples:
        print(f"[FastText-Jieba] 样本数不足 {cfg.min_samples}，当前={sample_count}，跳过训练")
        return
    
    # 加载样本（根据 profile 配置决定欠采样/全量）
    # 说明：
    # - balanced_undersample：欠采样随机平衡（每类<=max_samples/2）
    # - latest_full：平衡“全量模式”，每类取最新 max_samples/2
    # - random_full：平衡“全量随机模式”，每类随机抽 max_samples/2
    if cfg.sample_loading == SampleLoadingStrategy.latest_full:
        samples = storage.load_balanced_latest_samples(cfg.max_samples)
        print(f"[FastText-Jieba] 样本加载策略: latest_full (balanced latest)")
    elif cfg.sample_loading == SampleLoadingStrategy.random_full:
        samples = storage.load_balanced_random_samples(cfg.max_samples)
        print(f"[FastText-Jieba] 样本加载策略: random_full (balanced random)")
    else:
        samples = storage.load_balanced_samples(cfg.max_samples)
        print(f"[FastText-Jieba] 样本加载策略: balanced_undersample")

    print(f"[FastText-Jieba] 开始训练，共 {len(samples)} 个样本")
    
    # 统计标签分布
    label_counts = {}
    for sample in samples:
        label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
    print(f"[FastText-Jieba] 标签分布: 正常={label_counts.get(0, 0)}, 违规={label_counts.get(1, 0)}")
    
    # 准备训练数据文件（使用 jieba 分词）
    train_file = _prepare_training_file_jieba(samples, profile)
    
    try:
        # 训练模型
        print(f"[FastText-Jieba] 训练参数:")
        print(f"  维度: {cfg.dim}")
        print(f"  学习率: {cfg.lr}")
        print(f"  轮数: {cfg.epoch}")
        print(f"  词级 n-gram: {cfg.word_ngrams}")
        print(f"  子词 n-gram: 关闭 (使用 jieba 分词)")
        
        model = fasttext.train_supervised(
            input=train_file,
            dim=cfg.dim,
            lr=cfg.lr,
            epoch=cfg.epoch,
            wordNgrams=cfg.word_ngrams,
            minn=0,  # 关闭子词 n-gram
            maxn=0,  # 关闭子词 n-gram
            bucket=cfg.bucket,
            loss="ova",  # one-vs-all，适合二分类
            thread=2,  # 限制为单线程，降低 CPU 使用率
            verbose=2
        )
        
        # 保存模型
        model_path = profile.get_fasttext_model_path()
        model.save_model(model_path)
        print(f"[FastText-Jieba] 模型已保存: {model_path}")
        
        # 评估
        result = model.test(train_file)
        print(f"[FastText-Jieba] 训练集评估:")
        print(f"  样本数: {result[0]}")
        print(f"  准确率: {result[1]:.3f}")
        print(f"  召回率: {result[2]:.3f}")
        
    finally:
        # 清理临时文件
        if os.path.exists(train_file):
            os.remove(train_file)


def _prepare_training_file_jieba(samples, profile: ModerationProfile) -> str:
    """
    准备 fastText 训练文件（使用高级分词）
    
    格式: __label__0 词1 词2 词3 ...
          __label__1 词1 词2 词3 ...
    
    Returns:
        训练文件路径
    """
    cfg = profile.config.fasttext_training
    use_jieba = cfg.use_jieba
    use_tiktoken = cfg.use_tiktoken
    tiktoken_model = cfg.tiktoken_model
    
    # 创建临时文件
    fd, train_file = tempfile.mkstemp(suffix=".txt", prefix="fasttext_advanced_train_")
    
    # 确定分词模式描述
    if use_tiktoken and use_jieba:
        mode_desc = "tiktoken + jieba 组合分词"
    elif use_tiktoken:
        mode_desc = f"tiktoken 分词 (模型: {tiktoken_model})"
    elif use_jieba:
        mode_desc = "jieba 分词"
    else:
        mode_desc = "无分词（不应该到这里）"
    
    print(f"[FastText-Advanced] 开始分词: {mode_desc}")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        # 使用 tqdm 显示分词进度
        for sample in tqdm(samples, desc=mode_desc, unit="样本"):
            # 预处理文本
            text = sample.text.replace('\n', ' ').replace('\r', ' ')
            
            # 根据配置进行分词
            segmented_text = tokenize_text(text, use_jieba, use_tiktoken, tiktoken_model)
            
            # fastText 格式: __label__<类别> <分词后的文本>
            label = sample.label  # 0 或 1
            f.write(f"__label__{label} {segmented_text}\n")
    
    print(f"[FastText-Advanced] 训练文件已生成: {train_file}")
    return train_file


def _load_fasttext_with_cache(profile: ModerationProfile) -> fasttext.FastText:
    """
    加载 fastText 模型（带缓存）
    
    Returns:
        fastText 模型对象
    """
    profile_name = profile.profile_name
    model_path = profile.get_fasttext_model_path()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"fastText 模型不存在: {model_path}")
    
    # 获取文件修改时间
    model_mtime = os.path.getmtime(model_path)
    
    # 检查缓存
    if profile_name in _fasttext_cache:
        cached_model, cached_mtime = _fasttext_cache[profile_name]
        
        if model_mtime == cached_mtime:
            print(f"[DEBUG] 重用缓存的 fastText 模型: {profile_name}")
            return cached_model
        else:
            print(f"[DEBUG] fastText 模型文件已更新，重新加载: {profile_name}")
            del _fasttext_cache[profile_name]
    
    # 加载模型
    print(f"[DEBUG] 加载 fastText 模型: {model_path}")
    model = fasttext.load_model(model_path)
    
    # 保存到缓存
    _fasttext_cache[profile_name] = (model, model_mtime)
    
    return model


def fasttext_predict_proba_jieba(text: str, profile: ModerationProfile) -> float:
    """
    使用 fastText 模型预测违规概率（高级分词版本）
    
    Args:
        text: 待预测文本
        profile: 配置对象
        
    Returns:
        违规概率 (0-1)
    """
    cfg = profile.config.fasttext_training
    use_jieba = cfg.use_jieba
    use_tiktoken = cfg.use_tiktoken
    tiktoken_model = cfg.tiktoken_model
    
    # 确定分词模式
    if use_tiktoken and use_jieba:
        mode_desc = "tiktoken+jieba"
    elif use_tiktoken:
        mode_desc = "tiktoken"
    elif use_jieba:
        mode_desc = "jieba"
    else:
        mode_desc = "unknown"
    
    print(f"[DEBUG] fastText-Advanced 模型预测 (分词: {mode_desc})")
    
    # 加载模型（带缓存）
    model = _load_fasttext_with_cache(profile)
    
    # 预处理文本
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 根据配置进行分词
    segmented_text = tokenize_text(text, use_jieba, use_tiktoken, tiktoken_model)
    
    # 预测（返回 top-k 标签和概率）
    labels, probs = model.predict(segmented_text, k=2)
    
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