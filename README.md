# NumPy Neural Network Framework

基于 NumPy 从零实现的轻量级神经网络训练框架。

## 1. 项目简介

本项目为人工智能相关课程的小组实践项目，旨在使用 Python 和 NumPy 实现神经网络的基本组件，理解神经网络的前向传播、反向传播、梯度计算和参数优化过程。

项目不依赖 PyTorch、TensorFlow 等深度学习框架提供的神经网络层和自动求导功能，主要通过 NumPy 实现核心计算。

计划支持多层感知机（MLP）和卷积神经网络（CNN），并使用 MNIST 手写数字数据集进行训练与测试。

## 2. 项目功能

**基础神经网络模块**

- 全连接层（Linear）
- 激活函数（ReLU、Sigmoid、Tanh）
- 多层感知机（MLP）
- 二维卷积层（Conv2D）
- 简单卷积神经网络（CNN）

**待集成的训练功能**

- 损失函数及评价指标
- 优化器和训练流程
- 数据集加载与预处理
- MNIST 分类实验
- 训练结果可视化

## 3. 项目结构

```text
NeuralNetwork/
├── core/
│   ├── layers.py
│   ├── activations.py
│   └── conv.py
├── models/
│   ├── mlp.py
│   └── cnn.py
├── test_layers.py
├── test_activations.py
├── test_mlp.py
├── test_conv.py
└── test_cnn.py
```

以上为基础模块及 CNN 模块的目标结构。优化器、损失函数、数据处理和训练入口将在后续开发中集成。

## 4. 环境配置

建议使用 Python 3.11。

安装基本依赖：

```bash
pip install numpy matplotlib pytest
```

项目主要基于 NumPy 进行计算，无须配置 CUDA 或 GPU 环境。

## 5. 运行与测试

在项目根目录运行：

```bash
python test_layers.py
python test_activations.py
python test_mlp.py
python test_conv.py
python test_cnn.py
```

各测试文件分别用于验证神经网络基础层、激活函数、MLP、卷积层及 CNN 的基本功能。

完整模型训练流程将在后续集成完成后补充。

## 6. 小组分工

| 成员 | 负责模块 | 主要任务 |
|---|---|---|
| 成员 1 | 基础神经网络 | 全连接层、激活函数、MLP |
| 成员 2 | 训练与优化 | 优化器、训练流程 |
| 成员 3 | 卷积神经网络 | Conv2D、CNN |
| 成员 4 | 损失与评估 | 损失函数、评价指标 |
| 成员 5 | 数据处理 | 数据加载、预处理、预测 |
| 成员 6 | 项目集成 | 模块整合、实验与展示 |

## 7. 开发与协作

项目使用 GitHub 进行版本管理和多人协作。

建议各成员在独立功能分支中开发，完成模块测试后通过 Pull Request 合并到主分支，避免直接修改其他成员正在开发的代码。

## 8. 开发状态

项目正在开发中。

基础神经网络模块已由成员 1 完成初步实现，卷积层及 CNN 已提供初版实现和测试。其他模块将在后续逐步集成，最终完成 MNIST 分类实验。