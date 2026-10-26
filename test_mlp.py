import numpy as np
from models.mlp import MLP

np.random.seed(42)

model = MLP(
    input_size=4,
    hidden_sizes=[8, 6],
    output_size=3,
    activation="relu"
)

x = np.random.randn(5, 4)

output = model.forward(x)

print("========== 前向传播测试 ==========")
print("输入形状：", x.shape)
print("输出形状：", output.shape)

grad_output = np.ones_like(output)
grad_input = model.backward(grad_output)

print("\n========== 反向传播测试 ==========")
print("输入梯度形状：", grad_input.shape)

print("\n========== 参数梯度测试 ==========")

for i, (param, grad) in enumerate(model.parameters()):
    print(f"参数{i + 1}：")
    print("  参数形状：", param.shape)
    print("  梯度形状：", None if grad is None else grad.shape)
    print("  梯度是否存在：", grad is not None)

    assert grad is not None, f"参数{i + 1}的梯度没有计算出来"
    assert param.shape == grad.shape, f"参数{i + 1}与梯度的形状不一致"

assert output.shape == (5, 3)
assert grad_input.shape == (5, 4)

print("\n所有测试通过！")