# fastText 模型迁移指南

本文档说明如何从 BoW (Bag of Words) 模型迁移到 fastText 模型。

## 背景

**当前实现：**
- 使用 `jieba + TF-IDF + SGDClassifier/LogisticRegression` 的 BoW 模型
- 完整的样本采集、训练调度、工具链
- 三段式审核逻辑：30% AI 抽样 + 本地模型三段判断

**迁移目标：**
- 使用更轻量的 fastText 模型替代 BoW
- 保持三段式审核逻辑不变
- 降低 CPU 和内存占用

## 配置说明

### 1. 本地模型类型选择

在 `profile.json` 中配置 `local_model_type`:

```json
{
  "local_model_type": "bow",  // 可选: "bow" 或 "fasttext"
  "bow_training": { ... },
  "fasttext_training": { ... }
}
```

### 2. fastText 训练参数

```json
{
  "fasttext_training": {
    "min_samples": 200,           // 最小样本数
    "retrain_interval_minutes": 60, // 重训练间隔（分钟）
    "max_samples": 50000,         // 最大样本数
    "dim": 64,                    // 词向量维度
    "lr": 0.1,                    // 学习率
    "epoch": 5,                   // 训练轮数
    "word_ngrams": 2,             // 词级 n-gram
    "minn": 2,                    // 子词最小长度
    "maxn": 4,                    // 子词最大长度
    "bucket": 200000              // hash 词表大小
  }
}
```

**参数调优建议：**
- **dim**: 词向量维度，越大越精确但训练和预测越慢，建议 64-128
- **lr**: 学习率，建议 0.1-0.5
- **epoch**: 训练轮数，建议 5-10
- **word_ngrams**: 词级 n-gram，2 表示 unigram + bigram
- **minn/maxn**: 字符级 n-gram 范围，[2,4] 可以捕获中文字符特征
- **bucket**: hash 表大小，越大内存占用越大但冲突越少

## 使用方法

### 1. 训练 fastText 模型

```bash
# 训练指定 profile 的 fastText 模型
python tools/train_fasttext_model.py default

# 查看训练输出
# [FastText] 开始训练，共 1000 个样本
# [FastText] 训练参数:
#   维度: 64
#   学习率: 0.1
#   ...
# [FastText] 模型已保存: configs/mod_profiles/default/fasttext_model.bin
```

### 2. 测试 fastText 模型

```bash
# 命令行测试
python tools/test_fasttext_model.py default "测试文本"

# 交互模式
python tools/test_fasttext_model.py default
# 进入交互模式，输入文本进行测试（输入 'quit' 退出）
# 请输入文本: 
```

### 3. 切换模型类型

修改 `configs/mod_profiles/default/profile.json`:

```json
{
  "local_model_type": "fasttext"  // 从 "bow" 改为 "fasttext"
}
```

重启服务后生效。

## 迁移步骤

### 阶段一：Shadow 测试（推荐）

1. **保持 BoW 运行**
   ```json
   { "local_model_type": "bow" }
   ```

2. **训练 fastText 模型**
   ```bash
   python tools/train_fasttext_model.py default
   ```

3. **离线对比测试**
   使用测试工具对比两个模型的结果：
   ```bash
   # 测试 BoW
   python tools/test_bow_model.py default "测试文本"
   
   # 测试 fastText
   python tools/test_fasttext_model.py default "测试文本"
   ```

4. **批量评估**
   从数据库中抽取样本，对比两个模型的表现：
   - 准确率
   - 误判率/漏判率
   - 不确定区间比例（需要 AI 复核的比例）

### 阶段二：灰度切换

1. **部分 profile 切换**
   先在低流量的 profile 上切换到 fastText：
   ```json
   { "local_model_type": "fasttext" }
   ```

2. **监控指标**
   - 审核通过率/拒绝率变化
   - AI 审核调用比例变化
   - CPU/内存占用变化
   - 请求延时变化

3. **调整阈值（如需要）**
   根据监控结果调整：
   ```json
   {
     "probability": {
       "low_risk_threshold": 0.2,
       "high_risk_threshold": 0.8
     }
   }
   ```

### 阶段三：全量切换

所有 profile 切换到 fastText 后：
- 保留 BoW 相关代码作为备用
- 标记为 deprecated
- 后续大版本可移除

## 回滚策略

如果 fastText 出现问题，可以立即回滚：

```json
{
  "local_model_type": "bow"  // 切回 BoW
}
```

服务会自动加载 BoW 模型，无需重启。

## 性能对比

| 指标 | BoW | fastText | 说明 |
|------|-----|----------|------|
| 训练速度 | 慢 | 快 | fastText 无需 TF-IDF 计算 |
| 预测速度 | 中 | 快 | fastText 模型更轻量 |
| 内存占用 | 高 | 低 | fastText 不存储大的 TF-IDF 矩阵 |
| 准确率 | 高 | 中-高 | 需要根据数据调参 |

## 常见问题

### Q: fastText 模型文件在哪里？
A: `configs/mod_profiles/{profile}/fasttext_model.bin`

### Q: 如何查看当前使用的模型类型？
A: 查看日志输出：
```
[DEBUG] 决策路径: 本地模型预测 (类型=fasttext)
```

### Q: 两种模型可以同时存在吗？
A: 可以，切换模型类型后会自动加载对应模型。

### Q: fastText 训练失败怎么办？
A: 检查：
1. 样本数是否足够（min_samples）
2. fasttext 库是否已安装：`pip install fasttext`
3. 查看详细错误日志

## 依赖安装

```bash
# 安装 fastText
pip install fasttext

# 或在 requirements.txt 中添加
fasttext>=0.9.2
```

**重要提示：新版 fastText API**
- 新版 `fasttext.load_model()` 返回 `FastText` 对象（不再是旧版的 `WordVectorModel` 或 `SupervisedModel`）
- 新版 `fasttext.train_supervised()` 也返回 `FastText` 对象
- 本项目已兼容新版 API，所有类型注解已更新为 `fasttext.FastText`
- 如果遇到警告 "load_model does not return WordVectorModel or SupervisedModel any more"，可以忽略，这只是提示 API 变化

## 相关文件

- 模型实现: [`ai_proxy/moderation/smart/fasttext_model.py`](../ai_proxy/moderation/smart/fasttext_model.py)
- 配置管理: [`ai_proxy/moderation/smart/profile.py`](../ai_proxy/moderation/smart/profile.py)
- 统一接口: [`ai_proxy/moderation/smart/ai.py`](../ai_proxy/moderation/smart/ai.py)
- 调度器: [`ai_proxy/moderation/smart/scheduler.py`](../ai_proxy/moderation/smart/scheduler.py)
- 训练工具: [`tools/train_fasttext_model.py`](../tools/train_fasttext_model.py)
- 测试工具: [`tools/test_fasttext_model.py`](../tools/test_fasttext_model.py)