# fastText API 更新说明

## 概述

新版 fastText Python 库的 API 发生了变化，本文档说明这些变化以及我们的代码如何适配。

## API 变化详情

### 1. `load_model()` 返回类型变化

**旧版 API:**
```python
import fasttext

# 返回 WordVectorModel（词向量模型）或 SupervisedModel（监督学习模型）
model = fasttext.load_model('model.bin')
type(model)  # <class 'fasttext.WordVectorModel'> 或 <class 'fasttext.SupervisedModel'>
```

**新版 API:**
```python
import fasttext

# 统一返回 FastText 对象
model = fasttext.load_model('model.bin')
type(model)  # <class 'fasttext.FastText._FastText'>
```

### 2. `train_supervised()` 返回类型变化

**旧版 API:**
```python
model = fasttext.train_supervised(input='data.txt')
type(model)  # <class 'fasttext.SupervisedModel'>
```

**新版 API:**
```python
model = fasttext.train_supervised(input='data.txt')
type(model)  # <class 'fasttext.FastText._FastText'>
```

## 兼容性说明

### 我们的代码已完全兼容

本项目的代码已经更新以兼容新版 API：

1. **类型注解已更新**
   ```python
   # ai_proxy/moderation/smart/fasttext_model.py
   
   # 缓存类型注解
   _fasttext_cache: Dict[str, Tuple[fasttext.FastText, float]] = {}
   
   # 函数返回类型注解
   def _load_fasttext_with_cache(profile: ModerationProfile) -> fasttext.FastText:
       ...
   ```

2. **使用方式保持不变**
   
   新的 `FastText` 对象保持了与旧版相同的方法接口：
   ```python
   # 预测方法
   labels, probs = model.predict(text, k=2)  # ✅ 兼容
   
   # 测试方法
   result = model.test(test_file)  # ✅ 兼容
   
   # 保存方法
   model.save_model(path)  # ✅ 兼容
   ```

3. **文档已更新**
   
   相关文档中已添加 API 变化说明：
   - [`ai_proxy/moderation/smart/fasttext_model.py`](../ai_proxy/moderation/smart/fasttext_model.py) - 模块文档字符串
   - [`docs/FASTTEXT_MIGRATION.md`](FASTTEXT_MIGRATION.md) - 迁移指南

## 警告信息

如果你看到以下警告：

```
Warning: `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
```

**这是正常的**，只是 fastText 库提醒开发者 API 已经改变。我们的代码已经适配新 API，可以安全忽略此警告。

## 影响范围

### ✅ 无需修改的部分

- 所有预测逻辑 (`predict()`)
- 模型训练逻辑 (`train_supervised()`)
- 模型保存/加载 (`save_model()`, `load_model()`)
- 模型测试 (`test()`)

### ✅ 已完成的修改

- 类型注解从 `SupervisedModel` 更新为 `fasttext.FastText`
- 添加文档说明 API 变化
- 代码注释说明兼容性

## 升级建议

如果你需要升级 fastText 库：

```bash
# 升级到最新版本
pip install --upgrade fasttext

# 或指定版本
pip install fasttext>=0.9.2
```

升级后无需修改代码，一切都会正常工作。

## 参考资料

- [fastText 官方文档](https://fasttext.cc/)
- [fastText Python 库](https://github.com/facebookresearch/fastText/tree/main/python)
- 本项目相关文件：
  - [`ai_proxy/moderation/smart/fasttext_model.py`](../ai_proxy/moderation/smart/fasttext_model.py)
  - [`docs/FASTTEXT_MIGRATION.md`](FASTTEXT_MIGRATION.md)
  - [`tools/train_fasttext_model.py`](../tools/train_fasttext_model.py)
  - [`tools/test_fasttext_model.py`](../tools/test_fasttext_model.py)

## 总结

- ✅ 新版 API 返回统一的 `FastText` 对象
- ✅ 我们的代码已完全兼容
- ✅ 方法接口保持不变
- ✅ 可以安全升级 fastText 库
- ℹ️ 警告信息可以忽略