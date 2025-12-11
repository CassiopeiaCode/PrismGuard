# 手动训练指南

## 快速开始

### 训练 BoW 模型
```bash
python tools/train_bow_model.py default
```

### 训练 fastText 模型
```bash
python tools/train_fasttext_model.py default
```

> 将 `default` 替换为你的 profile 名称

## 详细步骤

### 1. 检查样本数据

训练前先检查样本数量是否足够：

```bash
# 使用诊断工具查看数据统计
python tools/diagnose_training_data.py default
```

输出示例：
```
[诊断] 配置: default
[诊断] 数据库: configs/mod_profiles/default/history.db
[诊断] 样本总数: 1234
[诊断] 标签分布:
  通过 (0): 800 (64.8%)
  违规 (1): 434 (35.2%)
```

### 2. 训练 BoW 模型

```bash
python tools/train_bow_model.py default
```

**预期输出：**
```
============================================================
BoW 模型训练工具
配置: default
============================================================

训练配置:
  最小样本数: 30
  最大样本数: 500
  最大特征数: 5000
  ...

样本统计:
  总数: 1234
  通过: 800
  违规: 434

开始训练...

[BOW] 开始训练，共 500 个样本
[BOW] 分层词表构建完成: 选中 5000 个特征
[BOW] 训练完成
[BOW] 训练准确率: 0.987

✅ 训练完成
模型已保存: configs/mod_profiles/default/bow_model.pkl
向量化器已保存: configs/mod_profiles/default/bow_vectorizer.pkl
```

### 3. 训练 fastText 模型

```bash
python tools/train_fasttext_model.py default
```

**预期输出：**
```
============================================================
fastText 模型训练工具
配置: default
============================================================

训练配置:
  最小样本数: 30
  最大样本数: 500
  维度: 50
  学习率: 0.1
  训练轮数: 5
  ...

样本统计:
  总数: 1234
  通过: 800
  违规: 434

开始训练...

[FastText] 开始训练，共 500 个样本
[FastText] 训练参数:
  维度: 50
  学习率: 0.1
  轮数: 5
  词级 n-gram: 2
  字符级 n-gram: [2, 4]

Progress: 100.0% words/sec/thread: 12345 lr: 0.00000 avg.loss: 0.12345
[FastText] 模型已保存: configs/mod_profiles/default/fasttext_model.bin
[FastText] 训练集评估:
  样本数: 500
  准确率: 0.982
  召回率: 0.976

✅ 训练完成
模型已保存: configs/mod_profiles/default/fasttext_model.bin
```

### 4. 测试模型

#### 测试 BoW 模型
```bash
# 单次测试
python tools/test_bow_model.py default "这是一条测试文本"

# 交互模式
python tools/test_bow_model.py default
```

#### 测试 fastText 模型
```bash
# 单次测试
python tools/test_fasttext_model.py default "这是一条测试文本"

# 交互模式
python tools/test_fasttext_model.py default
```

**预期输出：**
```
============================================================
fastText 模型测试工具
配置: default
============================================================

阈值配置:
  低风险阈值: 0.15
  高风险阈值: 0.85

测试文本: 这是一条测试文本
------------------------------------------------------------
[DEBUG] fastText 模型预测
  预测标签: ('__label__0',)
  预测概率: [0.9876]
  违规概率: 0.012

违规概率: 0.012
预测结果: ✅ 通过（低风险）
决策: 直接放行
```

## 常见问题

### Q1: 训练失败：样本数不足

**错误信息：**
```
[FastText] 样本数不足 30，当前=15，跳过训练
```

**解决方法：**
1. 等待系统自动采集更多样本（通过 AI 审核）
2. 降低 `min_samples` 配置（不推荐）

### Q2: 训练失败：模块未安装

**错误信息：**
```
ModuleNotFoundError: No module named 'fasttext'
```

**解决方法：**
```bash
pip install fasttext
```

### Q3: 如何强制重新训练？

训练工具会自动覆盖现有模型，无需额外操作：
```bash
# 直接运行即可覆盖
python tools/train_fasttext_model.py default
```

### Q4: 如何查看当前使用的模型？

检查配置文件中的 `local_model_type`：
```bash
# Linux/Mac
cat configs/mod_profiles/default/profile.json | grep local_model_type

# Windows
type configs\mod_profiles\default\profile.json | findstr local_model_type
```

输出：
```json
"local_model_type": "bow"  // 或 "fasttext"
```

### Q5: 训练需要多长时间？

**取决于样本数量：**
- 500 样本 + BoW: 5-10 秒
- 500 样本 + fastText: 2-5 秒
- 10000 样本 + BoW: 30-60 秒
- 10000 样本 + fastText: 10-20 秒

fastText 通常比 BoW 快 2-3 倍。

## 自动训练

系统会自动定时训练，无需手动操作：

**触发条件：**
1. 样本数 >= `min_samples`
2. 距离上次训练时间 >= `retrain_interval_minutes`

**检查间隔：**
- 默认每 10 分钟检查一次（由启动参数控制）

**查看日志：**
```
[SCHEDULER] 扫描到 1 个配置: default
[SCHEDULER] 开始训练: default (模型类型=bow)
[BOW] 开始训练，共 500 个样本
[BOW] 训练完成
[SCHEDULER] 训练完成: default
```

## 批量训练

如果有多个 profile，可以批量训练：

```bash
# 训练所有 profile 的 BoW 模型
for profile in default profile1 profile2; do
    python tools/train_bow_model.py $profile
done

# 训练所有 profile 的 fastText 模型
for profile in default profile1 profile2; do
    python tools/train_fasttext_model.py $profile
done
```

## 对比测试

对比两个模型在相同文本上的表现：

```bash
# 准备测试文本
echo "测试违规文本内容" > test.txt

# 测试 BoW
python tools/test_bow_model.py default "$(cat test.txt)"

# 测试 fastText
python tools/test_fasttext_model.py default "$(cat test.txt)"

# 对比输出的违规概率
```

## 训练后的验证

训练完成后，建议验证以下几点：

### 1. 检查模型文件是否存在

**BoW:**
```bash
ls -lh configs/mod_profiles/default/bow_model.pkl
ls -lh configs/mod_profiles/default/bow_vectorizer.pkl
```

**fastText:**
```bash
ls -lh configs/mod_profiles/default/fasttext_model.bin
```

### 2. 验证模型可用性

尝试进行一次预测：
```bash
python tools/test_fasttext_model.py default "测试"
```

如果输出正常的概率值，说明模型可用。

### 3. 对比训练前后的效果

记录训练前后的准确率变化，确保新模型不劣于旧模型。

## 高级用法

### 查看详细训练日志

在训练脚本中已经包含了详细的日志输出，无需额外配置。

### 自定义训练参数

临时修改配置文件后训练：
```bash
# 1. 备份原配置
cp profile.json profile.json.backup

# 2. 修改配置（如增加 epoch）
# 编辑 profile.json: "epoch": 10

# 3. 训练
python tools/train_fasttext_model.py default

# 4. 恢复配置
mv profile.json.backup profile.json
```

## 总结

**日常使用推荐：**
1. 让系统自动训练（最省心）
2. 定期检查训练日志
3. 只在需要时手动训练（如配置变更后）

**手动训练场景：**
- 修改了训练参数，想立即生效
- 切换模型类型后，首次训练
- 调试和测试新配置