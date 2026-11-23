# 成员2：经典感知机与训练优化模块（main 目录版本）

按最终目录结构放置：`models/perceptron.py`、`training/optimizer.py`、`training/trainer.py`、`tests/`。

## 运行

```bash
python -m pip install numpy pytest
python perceptron_demo.py
python -m pytest tests/test_member2.py tests/test_perceptron.py -v
```

## 接口说明

- `Perceptron(...).fit(X, y).predict(X)`：经典误分类驱动的感知机，不使用梯度反传；仅作为独立模型演示。
- `SGD(model, lr=0.01)` / `SGD(model, lr=0.01, momentum=0.9)` / `Adam(model, lr=0.001)`。
- `training.trainer.fit(model, train_loader, validation_loader, loss_fn, epochs, optimizer)` 使用成员6的 BatchLoader 训练 MLP/CNN；`evaluate(model, loader, loss_fn)` 返回指标字典。
- 优化器每次 `step()` 会重新读取 `model.parameters()`，兼容现有 `(参数, 梯度)` 接口。
- 已有 `core/losses.py` 由成员4负责，`trainer.py` 直接调用，不修改原文件。

## 与成员6分支合并（必须注意）

成员6已有仓库根目录 `training.py`，并使用 `from training import SGD, fit, evaluate`。
本交付采用 `training/` 包；后续同时存在 `training.py` 时 Python 导入优先级可能引发冲突。
本包已经将成员6现有的 `fit/evaluate` 行为适配至 `training/trainer.py`，并由 `training/__init__.py` 暴露接口。整合时应删除/停用根目录 `training.py`，让 `from training import SGD, fit, evaluate` 统一指向 `training/` 包；再次运行完整集成测试。

本文件包以当前 `main` 分支目录为空的 `training/` 包为基础；如需保留成员6完整功能，请在合并前完成上述协调。

## GitHub main 提交

网页上传时选择 `main`，上传压缩包中的各个文件（不要上传 zip 本身）；确认路径精确匹配。
另需在最终课程设计报告里如实声明 AI 工具参与情况。
