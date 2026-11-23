# NumPy Neural Network Framework

基于 Python 和 NumPy 手写前向传播、反向传播和参数更新的 MNIST 神经网络课程项目，支持 MLP 与 CNN。

现在可以通过统一入口完成：训练、独立验证、测试集评估、模型保存与重载、Loss/Accuracy 曲线、预测图片展示。核心计算不依赖 PyTorch 或 TensorFlow。

## 安装

建议使用 Python 3.11；本地也已在 Python 3.9 环境验证。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

首次运行会自动下载 MNIST 到 `data/`。运行产物保存在 `results/`；数据、环境和模型文件均不提交到 Git。

## 运行完整实验

在项目根目录执行：

```bash
python main.py train --config configs/mlp.json
python main.py train --config configs/cnn.json
```

两份配置都使用原始训练集中的 55,000 张图片训练、5,000 张图片验证，最后在官方 10,000 张测试图片上评估。验证集按固定种子划分，与训练集互不重叠。测试集不参与训练、曲线记录或参数选择。

MLP 默认结构为 `784 → 128 → 64 → 10`，训练 5 轮；CNN 为 `Conv2D(1→4, 3×3) → ReLU → Flatten → Linear(3136→10)`，训练 3 轮。默认均使用基础 SGD 和 logits 交叉熵。

每次训练输出以下文件到 `results/mlp/` 或 `results/cnn/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 实际使用的完整配置 |
| `history.csv` | 每轮训练/验证 Loss 和 Accuracy |
| `metrics.json` | 样本数量、测试 Loss、Accuracy、宏平均 Precision/Recall/F1 |
| `model.npz` | 最后一轮权重和重建模型所需配置 |
| `curves.png` | 训练/验证曲线 |
| `predictions.png` | 测试图片、真实标签和预测标签 |

训练曲线中的训练指标按当轮各批次更新前的输出累计，验证指标在当轮更新结束后计算。相同输出目录会被后续运行覆盖；对比实验请指定不同的 `--output-dir`。

## 已运行的整合基线

两份默认配置已于 2026-11-23 实际跑完，均评估官方 10,000 张测试图片：MLP 训练 5 轮，Accuracy 为 **96.31%**；CNN 训练 3 轮，Accuracy 为 **91.90%**。模型结构和训练参数不同，正式对比方案由小组统一。

实际配置、完整指标、逐轮记录与图片见 [基线实验结果](examples/baseline/README.md)。

## 先做小规模联调

```bash
python main.py train --config configs/mlp.json --epochs 2 --train-limit 2000 --validation-size 500 --test-limit 500 --output-dir results/smoke-mlp
python main.py train --config configs/cnn.json --epochs 2 --train-limit 1000 --validation-size 500 --test-limit 500 --output-dir results/smoke-cnn
```

命令行参数覆盖配置文件。可调整隐藏层、激活函数、卷积参数、学习率、轮数、批次大小、随机种子和损失：

```bash
python main.py train --config configs/mlp.json --hidden-sizes 256 128 64 --activation tanh --learning-rate 0.01 --loss mse --output-dir results/mlp-mse
python main.py train --help
```

分类标签始终为整数 `(N,)`。交叉熵直接接收模型 logits；选择 MSE 时，训练框架在损失边界转换为 one-hot 标签。

## 重载、评估和预测

```bash
python main.py evaluate --checkpoint results/mlp/model.npz --output results/mlp/reloaded-metrics.json
python main.py predict --checkpoint results/mlp/model.npz --num-samples 5 --output results/mlp/reloaded-predictions.png
python main.py predict --checkpoint results/cnn/model.npz --num-samples 1 --output results/cnn/single-prediction.png
```

检查点包含模型结构，重载时无需手动重填隐藏层或卷积参数。默认只保存图片，适合终端和 CI；需要弹窗时，为 `train` 或 `predict` 加 `--show`。

## 测试

```bash
python -m pytest -q
```

覆盖损失与指标、数值梯度、卷积层、完整 CNN、MLP/CNN 的统一训练、两种损失、检查点重载、单张预测以及命令行训练/评估/预测。命令行测试使用本地生成的 IDX 图片，不下载 MNIST。

成员 4 的原有独立测试入口和基础模块脚本仍可单独运行：

```bash
python run_tests.py
python test_layers.py
python test_activations.py
python test_mlp.py
python test_conv.py
python test_cnn.py
```

GitHub Actions 在推送和 Pull Request 时使用 Python 3.11 运行测试。

## 目录与分工

```text
NeuralNetwork/
├── main.py                    # 成员 6：命令行入口
├── config.py / configs/       # 成员 6：配置
├── experiment.py              # 成员 6：实验组织、数据划分和输出
├── training.py                # 成员 6：统一训练接口与基础 SGD
├── core/                      # 成员 1/3/4：层、激活、卷积、损失
├── models/                    # 成员 1/3：MLP、CNN；factory.py 为成员 6 适配
├── utils/
│   ├── dataset.py             # 成员 5：MNIST、批次加载
│   ├── predict.py             # 成员 5：参数读写、预测；成员 6 衔接与单张展示修复
│   ├── checkpoint.py          # 成员 6：结构与权重恢复
│   ├── visualization.py       # 成员 6：训练曲线
│   ├── metrics.py             # 成员 4：分类指标
│   └── gradient_check.py      # 成员 4：数值梯度
├── tests/                     # 模块与整合测试
└── docs/INTEGRATION.md         # 小组接口与最终交付清单
```

成员 2 原定的经典感知机和正式训练/优化模块尚待提交。当前基础 SGD 已让整合流程可以运行，后续算法沿现有 `step()` 接口接入；具体协作安排见 [整合交接](docs/INTEGRATION.md)。成员 4 的模块说明见 [README_PART4.md](README_PART4.md)。

建议各成员在独立分支开发并通过 Pull Request 合并，保留各自模块的实现和报告贡献。
