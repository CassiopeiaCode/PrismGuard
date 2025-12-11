"""
fastText 模型训练和预测模块（jieba 分词版本）
使用 jieba 分词 + fastText 进行文本分类

相比原版的改进：
- 使用 jieba 分词，更符合中文语言特性
- 关闭子词 n-gram (minn=0, maxn=0)
- 使用词级 n-gram 提取特征
- 使用 tqdm 显示分词进度
"""
import os
import tempfile
import fasttext
import jieba
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.storage import SampleStorage


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
    
    # 加载样本
    samples = storage.load_balanced_samples(cfg.max_samples)
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
    准备 fastText 训练文件（使用 jieba 分词）
    
    格式: __label__0 词1 词2 词3 ...
          __label__1 词1 词2 词3 ...
    
    Returns:
        训练文件路径
    """
    # 创建临时文件
    fd, train_file = tempfile.mkstemp(suffix=".txt", prefix="fasttext_jieba_train_")
    
    print(f"[FastText-Jieba] 开始 jieba 分词...")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        # 使用 tqdm 显示分词进度
        for sample in tqdm(samples, desc="jieba 分词", unit="样本"):
            # 预处理文本
            text = sample.text.replace('\n', ' ').replace('\r', ' ')
            
            # jieba 分词
            words = jieba.cut(text)
            segmented_text = ' '.join(words)
            
            # fastText 格式: __label__<类别> <分词后的文本>
            label = sample.label  # 0 或 1
            f.write(f"__label__{label} {segmented_text}\n")
    
    print(f"[FastText-Jieba] 训练文件已生成: {train_file}")
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
    使用 fastText 模型预测违规概率（jieba 分词版本）
    
    Args:
        text: 待预测文本
        profile: 配置对象
        
    Returns:
        违规概率 (0-1)
    """
    print(f"[DEBUG] fastText-Jieba 模型预测")
    
    # 加载模型（带缓存）
    model = _load_fasttext_with_cache(profile)
    
    # 预处理文本
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # jieba 分词
    words = jieba.cut(text)
    segmented_text = ' '.join(words)
    
    # 预测（返回 top-k 标签和概率）
    labels, probs = model.predict(segmented_text, k=2)
    
    print(f"  预测标签: {labels}")
    print(f"  预测概率: {probs}")
    
    # 找出"违规"标签（__label__1）的概率
    violation_prob = 0.0
    for label, p in zip(labels, probs):
        if label == "__label__1":
            violation_prob = float(p)
            break
    
    print(f"  违规概率: {violation_prob:.3f}")
    return violation_prob


def fasttext_model_exists(profile: ModerationProfile) -> bool:
    """检查 fastText 模型是否存在"""
    return os.path.exists(profile.get_fasttext_model_path())