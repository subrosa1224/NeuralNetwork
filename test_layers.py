import numpy as np
from core.layers import Linear


np.random.seed(42)

layer = Linear(input_size=3, output_size=2)

x = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

output = layer.forward(x)

print("输入数据：")
print(x)

print("\n前向传播输出：")
print(output)

grad_output = np.ones_like(output)

grad_input = layer.backward(grad_output)

print("\n输入梯度：")
print(grad_input)

print("\n权重梯度形状：")
print(layer.dW.shape)

print("\n偏置梯度形状：")
print(layer.db.shape)

print("\n测试完成！")