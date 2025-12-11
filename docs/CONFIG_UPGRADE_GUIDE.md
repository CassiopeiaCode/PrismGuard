# 配置升级指南：添加 fastText 支持

## 升级说明

你的原配置已经非常完善，只需要添加两个新字段即可支持 fastText 迁移：

1. `local_model_type`: 指定使用的本地模型类型
2. `fasttext_training`: fastText 训练配置

## 升级前后对比

### 原配置保持不变
```json
{
  "ai": { ... },              // ✅ 保持不变
  "prompt": { ... },          // ✅ 保持不变
  "probability": { ... },     // ✅ 保持不变
  "bow_training": { ... }     // ✅ 保持不变
}
```

### 新增配置
```json
{
  // ... 原有配置 ...
  
  "local_model_type": "bow",  // ⭐ 新增：模型类型选择
  
  "fasttext_training": {      // ⭐ 新增：fastText 训练配置
    "min_samples": 30,
    "retrain_interval_minutes": 600,
    "max_samples": 500,
    "dim": 50,
    "lr": 0.1,
    "epoch": 5,
    "word_ngrams": 2,
    "minn": 2,
    "maxn": 4,
    "bucket": 100000
  }
}
```

## 你的升级配置（已优化）

```json
{
  "ai": {
    "provider": "openai",
    "base_url": "http://100.64.0.101:40021/v1",
    "model": "gpt-4.1-nano",
    "api_key_env": "CLAUDE_MOD_API_KEY",
    "timeout": 15
  },
  "prompt": {
    "template_file": "ai_prompt.txt",
    "max_text_length": 60000
  },
  "probability": {
    "ai_review_rate": 0.0005,
    "random_seed": 42,
    "low_risk_threshold": 0.15,
    "high_risk_threshold": 0.85
  },
  "local_model_type": "bow",
  "bow_training": {
    "min_samples": 30,
    "retrain_interval_minutes": 600,
    "max_samples": 500,
    "max_features": 5000,
    "use_char_ngram": true,
    "char_ngram_range": [2, 3],
    "use_word_ngram": true,
    "word_ngram_range": [1, 2],
    "model_type": "sgd_logistic",
    "batch_size": 200,
    "max_seconds": 60,
    "max_db_items": 10000,
    "use_layered_vocab": true,
    "vocab_buckets": [
      {"name": "high_freq", "min_doc_ratio": 0.05, "max_doc_ratio": 0.6, "limit": 1200},
      {"name": "mid_freq", "min_doc_ratio": 0.01, "max_doc_ratio": 0.05, "limit": 2600},
      {"name": "low_freq", "min_doc_ratio": 0.002, "max_doc_ratio": 0.01, "limit": 1200}
    ]
  },
  "fasttext_training": {
    "min_samples": 30,
    "retrain_interval_minutes": 600,
    "max_samples": 500,
    "dim": 50,
    "lr": 0.1,
    "epoch": 5,
    "word_ngrams": 2,
    "minn": 2,
    "maxn": 4,
    "bucket": 100000
  }
}
```

## fastText 参数针对你场景的优化

根据你的配置特点（样本少、训练频繁），我对 fastText 参数做了优化：

### 1. 样本规模适配
```json
{
  "min_samples": 30,      // 与 BoW 保持一致
  "max_samples": 500      // 与 BoW 保持一致
}
```

### 2. 维度降低（减少过拟合）
```json
{
  "dim": 50               // 样本少时用 50 维足够（默认 64）
}
```

### 3. 词表大小优化
```json
{
  "bucket": 100000        // 样本少时 10 万就够（默认 20 万）
}
```

### 4. 其他参数保持标准
```json
{
  "lr": 0.1,              // 标准学习率
  "epoch": 5,             // 标准训练轮数
  "word_ngrams": 2,       // unigram + bigram
  "minn": 2,              // 字符 2-gram
  "maxn": 4               // 字符 4-gram
}
```

## 迁移步骤（保守策略）

### 第一步：升级配置（不影响现有服务）
```bash
# 1. 备份原配置
cp profile.json profile.json.backup

# 2. 使用升级后的配置
# 将上面的完整配置保存到 profile.json

# 3. 重启服务
# 服务会自动识别新配置，但仍使用 BoW（因为 local_model_type="bow"）
```

### 第二步：训练 fastText 模型
```bash
# 训练（不影响现有服务）
python tools/train_fasttext_model.py default

# 预期输出：
# [FastText] 开始训练，共 500 个样本
# [FastText] 训练参数:
#   维度: 50
#   学习率: 0.1
#   ...
# [FastText] 模型已保存
```

### 第三步：离线测试对比
```bash
# 测试 BoW
python tools/test_bow_model.py default "测试违规文本"

# 测试 fastText
python tools/test_fasttext_model.py default "测试违规文本"

# 对比结果，看概率是否接近
```

### 第四步：切换到 fastText（修改一个字段）
```json
{
  "local_model_type": "fasttext"  // 从 "bow" 改为 "fasttext"
}
```

### 第五步：监控指标
- 审核通过率/拒绝率是否变化
- CPU/内存占用是否降低
- 请求延时是否改善

### 第六步：如需回滚
```json
{
  "local_model_type": "bow"  // 改回 "bow" 即可
}
```

## 你的场景特点分析

根据你的配置，我注意到：

1. **极低的 AI 审核率** (`ai_review_rate: 0.0005`)
   - 说明非常依赖本地模型
   - fastText 的轻量化优势会更明显

2. **较少的样本量** (`max_samples: 500`)
   - 训练速度会很快（秒级）
   - BoW 和 fastText 都能应对

3. **长训练间隔** (`retrain_interval_minutes: 600`)
   - 10 小时才重训练一次
   - 训练速度不是瓶颈

4. **低阈值** (`low_risk_threshold: 0.15`)
   - 说明对误判容忍度低
   - 需要在测试中确认 fastText 的准确率

## 建议

### 优先级 1：先测试再切换
由于你的 `low_risk_threshold` 很低（0.15），说明对准确率要求高，建议：
1. 先用 `tools/test_fasttext_model.py` 充分测试
2. 确认准确率不劣于 BoW 再切换

### 优先级 2：如果资源充足，可以不迁移
你的配置中：
- 样本量不大（500）
- 训练间隔很长（10 小时）
- BoW 的性能影响可能不明显

如果 BoW 运行良好，可以保持现状，fastText 作为备选方案。

### 优先级 3：如果要降低资源占用
fastText 的优势：
- **内存更少**：不需要存储大的 TF-IDF 矩阵
- **预测更快**：模型更轻量

如果你的服务器资源紧张，或者希望降低每次预测的延时，fastText 是更好的选择。

## 完整配置文件

已保存到 `configs/mod_profiles/default/profile_upgraded.json`，你可以直接使用。