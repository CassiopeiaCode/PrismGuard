# NumPy 2.0 兼容性问题

## 问题描述

当使用 fastText 模型进行预测时，可能会遇到以下错误：

```
[DEBUG] 本地模型预测失败: Unable to avoid copy while creating an array as requested.
If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword. -> 回退到AI
```

## 原因分析

这是 **NumPy 2.0 兼容性问题**：

1. **NumPy 2.0 变更**: NumPy 2.0 弃用了 `np.array(obj, copy=False)` 语法
2. **fastText 未更新**: fastText 库内部仍在使用旧的 API
3. **自动回退**: 系统会自动捕获此错误并回退到 AI 审核

## 解决方案

### 方案 1: 降级 NumPy（推荐）

```bash
pip install 'numpy<2.0'
```

这是最简单且最稳定的解决方案，因为 fastText 官方尚未完全支持 NumPy 2.0。

### 方案 2: 等待 fastText 更新

关注 fastText 官方仓库的更新：
- GitHub: https://github.com/facebookresearch/fastText
- 等待官方发布兼容 NumPy 2.0 的版本

### 方案 3: 使用 BoW 模型

如果不想降级 NumPy，可以切换到 BoW (Bag of Words) 模型：

在配置文件中设置：
```json
{
  "local_model_type": "bow"
}
```

BoW 模型使用纯 Python 实现，不依赖 NumPy，因此不受此问题影响。

## 系统行为

当遇到此错误时，系统会：

1. ✅ **自动捕获错误**: 不会导致服务崩溃
2. ✅ **打印警告信息**: 提示 NumPy 兼容性问题
3. ✅ **自动回退**: 使用 AI 审核代替本地模型
4. ✅ **记录详细日志**: 首次出现时打印完整堆栈信息

示例日志输出：
```
[DEBUG] 本地模型预测失败: Unable to avoid copy while creating an array...
[WARNING] 检测到 NumPy 2.0 兼容性问题!
[WARNING] fastText 库尚未完全兼容 NumPy 2.0
[WARNING] 建议降级 NumPy: pip install 'numpy<2.0'
[WARNING] 或等待 fastText 更新以支持 NumPy 2.0
[DEBUG] 回退到 AI 审核
```

## 性能影响

- **短期影响**: 所有请求都会使用 AI 审核，增加延迟和成本
- **长期影响**: 无法利用本地模型的快速预测能力
- **建议**: 尽快降级 NumPy 以恢复本地模型功能

## 验证方法

### 检查当前 NumPy 版本

```bash
python -c "import numpy; print(numpy.__version__)"
```

### 测试 fastText 模型

```bash
python tools/test_fasttext_model.py
```

如果看到 NumPy 警告，说明需要降级。

## 相关链接

- [NumPy 2.0 迁移指南](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [fastText GitHub Issues](https://github.com/facebookresearch/fastText/issues)
- [本项目 fastText 文档](./FASTTEXT_MIGRATION.md)

## 更新日志

- **2025-12-11**: 添加自动检测和警告功能
- **2025-12-11**: 文档创建