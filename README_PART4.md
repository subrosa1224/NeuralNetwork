# 成员 4 交付说明：损失函数与评价指标

对应课程分工中的 **成员 4：损失函数与评价指标**，任务包括：
MSE、交叉熵、损失函数梯度、Accuracy、Precision、Recall、F1，以及单元测试。

> 所有文件都只放在本工作区（`NeuralNetwork小组作业/`）中，**没有修改 GitHub 仓库**。
> 文件路径按照讨论中确定的“最终项目目录”组织，可直接复制进项目对应位置。

---

## 一、交付文件

| 工作区文件 | 建议放入项目的位置 | 说明 |
|---|---|---|
| `core/losses.py` | `core/losses.py` | `Loss` 基类、`MSELoss`、`CrossEntropyLoss`、`build_loss` |
| `utils/metrics.py` | `utils/metrics.py` | Accuracy、Precision、Recall、F1、混淆矩阵、分类报告 |
| `utils/gradient_check.py` | `utils/gradient_check.py`（新增） | 中心差分数值梯度检查工具 |
| `tests/test_losses.py` | `tests/test_losses.py` | 26 个损失函数测试 |
| `tests/test_metrics.py` | `tests/test_metrics.py` | 21 个评价指标测试 |
| `tests/test_gradient_check.py` | `tests/test_gradient_check.py` | 7 个梯度检查工具测试 |
| `run_tests.py` | 项目根目录 | 不依赖 pytest 的测试入口 |
| `part4_demo.py` | 项目根目录（可选） | 功能演示：损失、梯度检查、指标、迷你训练 |
| `README_PART4.md` | — | 本说明 |

共 **54 个单元测试**，全部通过。代码只依赖 NumPy；测试既可以用 `python run_tests.py`
运行，也可以用项目已有的 pytest 运行。

---

## 二、快速开始

在项目根目录（本工作区）执行：

```bash
# 1. 运行全部单元测试（不需要 pytest）
python run_tests.py

# 2. 如果已经安装 pytest（-p no:cacheprovider 可避免生成缓存目录）
python -m pytest tests -v -p no:cacheprovider

# 3. 演示脚本：损失、梯度检查、评价指标、迷你训练
python part4_demo.py
```

环境要求：Python 3.8+、NumPy（项目已有）。本机验证环境为 Python 3.13 + NumPy 2.1.3。

---

## 三、接口与用法

### 3.1 损失函数

接口与项目中的层保持一致：`forward` 计算损失，`backward` 返回对输入的梯度。

```python
import numpy as np
from core.losses import MSELoss, CrossEntropyLoss, build_loss

# 交叉熵（pred 默认是未经过 softmax 的 logits）
loss = CrossEntropyLoss()
value = loss.forward(pred, target)   # pred: (N, C)，target: (N,) 或 (N, C)
grad = loss.backward()               # 对 pred 的梯度，形状同 pred

# 均方误差
mse = MSELoss()
value = mse.forward(pred, target)    # pred、target 形状必须一致
grad = mse.backward()

# 按配置名创建（供 config.py / 训练入口使用）
loss = build_loss("cross_entropy")
loss = build_loss("mse", reduction="sum")
```

要点：

* **reduction**：`"mean"`（默认）、`"sum"`、`"none"`。
  交叉熵的均值除以的是**样本数 N**（与 PyTorch 一致），不是 `N * C`。
* **target 形式**：
  * 类别下标：`(N,)`，例如 `np.array([3, 0, 7])`；
  * one-hot / 软标签：`(N, C)`，每行是概率，可用于标签平滑；
  * 单样本时 `pred` 可以是 `(C,)`，`target` 可以是标量。
* **from_logits**：
  * 默认 `True`，输入是 logits，内部用 log-sum-exp 计算，数值稳定；
  * 如果模型末端已经做了 softmax，则用 `CrossEntropyLoss(from_logits=False)`，
    输入概率会被截断到 `[eps, +inf)`，避免 `log(0)`。
* **数值稳定性**：logits 达到 1000 量级时损失与梯度仍有限；极端情况下
  softmax 下溢为 0 也不会出现 `nan`。
* **梯度正确性**：`tests/test_losses.py` 中用中心差分逐元素验证了
  logits、概率、硬标签、软标签四种组合的梯度。

`Loss` 基类还提供 `last_loss` 属性，方便训练循环记录当前损失：

```python
loss.forward(logits, y)
print(loss.last_loss)
```

### 3.2 评价指标

```python
from utils.metrics import (
    calculate_accuracy, calculate_precision, calculate_recall, calculate_f1,
    confusion_matrix, calculate_metrics, classification_report,
)

accuracy = calculate_accuracy(y_true, y_pred)
precision = calculate_precision(y_true, y_pred)              # 默认 macro
recall = calculate_recall(y_true, y_pred)
f1 = calculate_f1(y_true, y_pred)

# 二分类正类（标签 1）的指标
f1_positive = calculate_f1(y_true, y_pred, average="binary")

# 一次性返回字典，适合训练日志
metrics = calculate_metrics(y_true, y_pred)
# {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "support": ...}

# 混淆矩阵（行 = 真实类别，列 = 预测类别）与分类报告
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

要点：

* `y_pred` 既可以是类别下标 `(N,)`，也可以是模型输出 `(N, C)`（logits 或概率均可），
  二维输入会自动按行 `argmax`；`(N, 1)` 的整数标签列也支持；
* `y_true` 也接受 one-hot / 概率矩阵，以及 `(N, 1)` 的整数标签列；
* `average` 支持 `"macro"`（默认）、`"micro"`、`"weighted"`、`"binary"`、`None`；
  `None` 返回逐类数组，便于画图或做逐类分析；
* `labels=` 可以固定类别集合与顺序，例如 MNIST 用 `labels=range(10)`；
* 分母为 0（某一类没有样本或没有预测）时由 `zero_division` 控制，默认返回 0；
* 单标签多分类中 `average="micro"` 的 F1 等于 Accuracy，测试中已作为不变量验证。

---

## 四、与项目其他模块的对接

### 4.1 训练循环

`MLP.forward()` 默认 `output_activation=None`，输出的是 **logits**，因此默认的
`CrossEntropyLoss()` 可以直接使用：

```python
from core.losses import CrossEntropyLoss
from utils.metrics import calculate_accuracy, calculate_metrics

loss_fn = CrossEntropyLoss()
for epoch in range(epochs):
    for x_batch, y_batch in loader:
        logits = model.forward(x_batch)
        loss_value = loss_fn.forward(logits, y_batch)
        grad = loss_fn.backward()          # dL/dlogits
        model.backward(grad)               # 反向传播
        optimizer.step()                   # 参数更新

    # 每个 epoch 结束后评估
    logits = model.forward(x_val)
    acc = calculate_accuracy(y_val, logits)
    print(calculate_metrics(y_val, logits))
```

如果模型末端已经对 logits 做了 softmax（例如成员 1 在输出层新增 softmax 激活），
则改为 `CrossEntropyLoss(from_logits=False)`，两种写法在数值上等价（演示脚本中有一致性输出）。

配置化调用（对应成员 6 的 `config.py`）：

```python
config = {"loss": "cross_entropy", ...}
loss_fn = build_loss(config["loss"])
```

### 4.2 文件放置

* `core/losses.py`：与 `core/layers.py`、`core/activations.py`、`core/conv.py` 同级；
* `utils/metrics.py`：按最终项目目录新建 `utils/` 放入（`dataset.py`、`predict.py`、
  `visualization.py` 也在这个目录）；
* 测试文件：最终目录建议放在 `tests/`。仓库当前把 `test_*.py` 放在根目录，
  两种方式都可以，放入 `tests/` 后把 `run_tests.py` 一起提交即可。

导入方式与仓库现有代码一致（`from core.layers import Linear` 这种绝对导入），
不依赖相对导入，复制到项目中不需要改动。

---

## 五、单元测试覆盖

所有期望值都是**手算 / 公式直接计算**得到的，没有依赖 sklearn，因此不需要额外安装库。

| 测试文件 | 覆盖内容 |
|---|---|
| `tests/test_losses.py` | MSE 的 mean/sum/none 数值与梯度、零损失、上游梯度缩放、数值梯度检查、形状错误；CE 的 logits、概率、硬标签、软标签、one-hot 等价、单样本、大 logits 稳定性、softmax 下溢、三种归约、均值除以 N、重复 forward、未 forward 就 backward、非法标签/形状、`build_loss` |
| `tests/test_metrics.py` | 二分类与多分类的 Accuracy、Precision、Recall、F1；逐类 / macro / weighted / micro；混淆矩阵方向与自定义标签顺序；perfect prediction；zero_division；one-hot 真值；`(N, 1)` 标签列；分数矩阵输入等价性；micro-F1 = Accuracy；balanced 时 weighted = macro；F1 为调和平均；分类报告；空输入、长度不一致、非整数标签、重复/缺失 labels、非法 average |
| `tests/test_gradient_check.py` | 二次函数数值梯度、正确梯度通过、错误梯度被识别、形状不匹配、输入不被修改、相对误差公式、非法 eps |

运行结果（本机）：

```
54 个测试，通过 54 个，失败 0 个
```

---

## 六、数值梯度检查

`utils/gradient_check.py` 使用中心差分：

```python
from utils.gradient_check import check_gradient

def loss_of_logits(z):
    return loss.forward(z, y)

result = check_gradient(loss_of_logits, logits, loss.backward())
print(result["max_rel_error"])   # 一般在 1e-10 量级
```

返回字典包含 `ok`、`max_abs_error`、`max_rel_error`、`worst_index`、`shape`；
默认容差 `rtol=1e-4, atol=1e-6`，不通过时抛出 `AssertionError` 并打印最差位置。

该工具也可以用于检查成员 1 / 成员 3 的手写梯度：把“前向 + 标量损失”包成一个函数，
把待检查的梯度作为 `analytic_grad` 传入即可。建议用 `float64` 计算（项目现有 MLP
默认就是 `float64`），`eps=1e-6`；ReLU 在 0 附近不可导，应避开折点。

---

## 七、注意事项

1. **没有改动 GitHub 仓库**，也没有 clone 或写入其他目录；所有代码都在本工作区。
2. `average="binary"` 要求数据/`labels` 中包含正类 `1`；只有类别 0 时会明确报错，
   避免静默返回无意义的结果。
3. 交叉熵默认输入 logits；如果误传概率，程序不会报错，但结果没有意义。
   模型输出 softmax 时请显式使用 `from_logits=False`。
4. `reduction="none"` 时，交叉熵的 `backward(grad_output)` 支持逐样本上游梯度 `(N,)`
   （与 PyTorch 一致），也支持能与 `(N, C)` 广播的数组。
5. 测试输出为中文，与仓库现有 `test_*.py` 的风格一致；如终端编码异常，
   可设置 `PYTHONIOENCODING=utf-8`。

---

## 八、任务对照

| 任务要求 | 实现位置 | 验证方式 |
|---|---|---|
| 1. 手动实现 MSE | `core/losses.py::MSELoss` | 手算数值 + 数值梯度 |
| 2. 手动实现交叉熵 | `core/losses.py::CrossEntropyLoss` | 手算数值 + log-sum-exp 稳定性测试 |
| 3. 实现损失函数梯度 | 两个类的 `backward()` | 中心差分逐元素对比 |
| 4. 实现 Accuracy | `utils/metrics.py::calculate_accuracy` | 二分类/多分类/分数矩阵 |
| 5. 实现 Precision、Recall、F1 | `calculate_precision` / `calculate_recall` / `calculate_f1` | 手算混淆矩阵 + 三种平均 |
| 6. 编写单元测试 | `tests/` + `run_tests.py` | 54 个测试全部通过 |
