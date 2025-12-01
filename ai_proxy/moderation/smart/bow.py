"""
词袋模型（BoW）训练和预测模块
使用 jieba 分词 + TF-IDF + SGDClassifier
支持大规模数据的增量训练
"""
import os
import time
import random
import jieba
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from ai_proxy.moderation.smart.profile import ModerationProfile
from ai_proxy.moderation.smart.storage import SampleStorage
from ai_proxy.moderation.smart.ai import ModerationResult


# 模型缓存：{profile_name: (vectorizer, clf, model_mtime, vectorizer_mtime)}
_model_cache: Dict[str, Tuple[object, object, float, float]] = {}


def tokenize_for_bow(text: str, use_char_ngram: bool = True) -> str:
    """
    文本预处理和分词
    混合词级分词 + 字符级 n-gram
    
    Args:
        text: 原始文本
        use_char_ngram: 是否使用字符级 n-gram
        
    Returns:
        空格分隔的 token 序列
    """
    # 词级分词
    word_tokens = list(jieba.cut(text))
    
    tokens = word_tokens
    
    # 字符级 bigram（可选）
    if use_char_ngram:
        char_bigrams = [text[i:i+2] for i in range(len(text)-1)]
        char_trigrams = [text[i:i+3] for i in range(len(text)-2)]
        tokens.extend(char_bigrams)
        tokens.extend(char_trigrams)
    
    return " ".join(tokens)


def train_bow_model(profile: ModerationProfile):
    """
    训练词袋线性模型（一次性训练版本）
    """
    storage = SampleStorage(profile.get_db_path())
    cfg = profile.config.bow_training
    
    # 数据库清理
    storage.cleanup_excess_samples(cfg.max_db_items)
    
    # 检查样本数量
    sample_count = storage.get_sample_count()
    if sample_count < cfg.min_samples:
        print(f"[BOW] 样本数不足 {cfg.min_samples}，当前={sample_count}，跳过训练")
        return
    
    # 加载样本
    samples = storage.load_samples(cfg.max_samples)
    texts = [s.text for s in samples]
    labels = [s.label for s in samples]
    
    print(f"[BOW] 开始训练，共 {len(samples)} 个样本")
    
    # 文本预处理和分词
    use_char_ngram = cfg.use_char_ngram
    corpus = [tokenize_for_bow(t, use_char_ngram) for t in texts]
    
    # 构建 TF-IDF 向量化器
    word_ngram = cfg.word_ngram_range
    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=tuple(word_ngram) if cfg.use_word_ngram else (1, 1),
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(corpus)
    
    # 训练线性模型
    model_type = cfg.model_type
    
    if model_type == "sgd_logistic":
        clf = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            n_jobs=1,
            random_state=42
        )
    else:
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=1,
            random_state=42
        )
    
    clf.fit(X, labels)
    
    # 保存模型
    joblib.dump(vectorizer, profile.get_vectorizer_path())
    joblib.dump(clf, profile.get_model_path())
    
    print(f"[BOW] 训练完成")
    
    # 简单评估
    train_acc = clf.score(X, labels)
    print(f"[BOW] 训练准确率: {train_acc:.3f}")


def bow_model_exists(profile: ModerationProfile) -> bool:
    """检查词袋模型是否存在"""
    return (os.path.exists(profile.get_model_path()) and 
            os.path.exists(profile.get_vectorizer_path()))


def _load_model_with_cache(profile: ModerationProfile) -> Tuple[object, object]:
    """
    加载模型（带缓存，避免重复加载和内存泄漏）
    
    Returns:
        (vectorizer, clf)
    """
    profile_name = profile.profile_name
    model_path = profile.get_model_path()
    vectorizer_path = profile.get_vectorizer_path()
    
    # 获取文件修改时间
    model_mtime = os.path.getmtime(model_path)
    vectorizer_mtime = os.path.getmtime(vectorizer_path)
    
    # 检查缓存
    if profile_name in _model_cache:
        cached_vec, cached_clf, cached_model_mtime, cached_vec_mtime = _model_cache[profile_name]
        
        # 如果文件没有更新，重用缓存
        if model_mtime == cached_model_mtime and vectorizer_mtime == cached_vec_mtime:
            print(f"[DEBUG] 重用缓存的模型: {profile_name}")
            return cached_vec, cached_clf
        else:
            print(f"[DEBUG] 模型文件已更新，重新加载: {profile_name}")
            # 清理旧模型（帮助GC回收内存）
            del _model_cache[profile_name]
    
    # 加载模型
    print(f"[DEBUG] 加载模型文件: {profile_name}")
    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    
    # 保存到缓存
    _model_cache[profile_name] = (vectorizer, clf, model_mtime, vectorizer_mtime)
    
    return vectorizer, clf


def bow_predict_proba(text: str, profile: ModerationProfile) -> float:
    """
    使用词袋模型预测违规概率
    
    Args:
        text: 待预测文本
        profile: 配置
        
    Returns:
        违规概率 (0-1)
    """
    print(f"[DEBUG] 词袋模型预测")
    
    # 加载模型（带缓存）
    vectorizer, clf = _load_model_with_cache(profile)
    
    print(f"  模型类型: {type(clf).__name__}")
    print(f"  特征数量: {len(vectorizer.get_feature_names_out())}")
    
    # 预处理
    use_char_ngram = profile.config.bow_training.use_char_ngram
    corpus = [tokenize_for_bow(text, use_char_ngram)]
    X = vectorizer.transform(corpus)
    
    print(f"  文本特征维度: {X.shape}")
    print(f"  非零特征数: {X.nnz}")
    
    # 预测概率
    if hasattr(clf, 'predict_proba'):
        # SGDClassifier(loss="log_loss") 和 LogisticRegression 都支持
        proba = clf.predict_proba(X)[0]
        
        # ✅ 关键修复：检查模型的实际类别顺序
        actual_classes = clf.classes_
        print(f"  模型类别顺序: {actual_classes}")
        print(f"  原始概率分布: {proba}")
        
        # 找到类别1（违规）在概率数组中的位置
        if 1 in actual_classes:
            violation_idx = list(actual_classes).index(1)
            violation_prob = float(proba[violation_idx])
            print(f"  违规类别索引: {violation_idx}")
            print(f"  违规概率: {violation_prob:.3f}")
            return violation_prob
        else:
            print(f"  警告：模型中没有类别1，返回默认值0")
            return 0.0
    else:
        # 如果模型不支持 predict_proba，使用 decision_function
        score = clf.decision_function(X)[0]
        print(f"  决策函数值: {score:.3f}")
        # 简单的 sigmoid 转换
        import math
        prob = 1.0 / (1.0 + math.exp(-score))
        print(f"  转换后概率: {prob:.3f}")
        return prob


def bow_predict(text: str, profile: ModerationProfile) -> ModerationResult:
    """
    使用词袋模型进行预测（返回完整结果）
    """
    proba = bow_predict_proba(text, profile)
    
    # 根据阈值判断
    low_t = profile.config.probability.low_risk_threshold
    high_t = profile.config.probability.high_risk_threshold
    
    if proba < low_t:
        violation = False
        reason = f"BoW model: low risk (p={proba:.3f} < {low_t})"
    elif proba > high_t:
        violation = True
        reason = f"BoW model: high risk (p={proba:.3f} > {high_t})"
    else:
        # 中间不确定，需要 AI 复核
        violation = False
        reason = f"BoW model: uncertain (p={proba:.3f}), needs AI review"
    
    return ModerationResult(
        violation=violation,
        category=None,
        reason=reason,
        source="bow_model",
        confidence=proba
    )