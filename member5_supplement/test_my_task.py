
import tempfile
from pathlib import Path

import numpy as np

from utils.dataset import get_mnist_data, BatchLoader
from utils.predict import save_model_params, load_model_params


def main():
    print("=" * 55)
    print("成员五：MNIST 数据处理与参数保存加载测试")
    print("=" * 55)

    # 1. 测试 MNIST 数据加载、归一化与展平
    print("\n[1] MNIST 数据加载测试")

    (X_train, y_train), (X_test, y_test) = get_mnist_data(
        data_dir="data",
        flatten=True,
        normalize=True
    )

    assert X_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 784)
    assert y_test.shape == (10000,)

    assert X_train.dtype == np.float32
    assert y_train.dtype == np.int64
    assert np.min(X_train) >= 0.0
    assert np.max(X_train) <= 1.0

    print("训练集图像形状:", X_train.shape)
    print("训练集标签形状:", y_train.shape)
    print("测试集图像形状:", X_test.shape)
    print("测试集标签形状:", y_test.shape)
    print("像素范围:", X_train.min(), "~", X_train.max())
    print("MNIST 数据加载与预处理测试通过！")

    # 2. 测试 Mini-batch 迭代器
    print("\n[2] Mini-batch 数据加载测试")

    loader = BatchLoader(
        X_train,
        y_train,
        batch_size=32,
        shuffle=False
    )

    batch_X, batch_y = next(iter(loader))

    assert batch_X.shape == (32, 784)
    assert batch_y.shape == (32,)

    print("Batch 图像形状:", batch_X.shape)
    print("Batch 标签形状:", batch_y.shape)
    print("Mini-batch 测试通过！")

    # 3. 测试模型参数保存与加载
    print("\n[3] 模型参数保存与加载测试")

    dummy_params = {
        "W1": np.array([[1, 2], [3, 4]]),
        "b1": np.array([0, 1])
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_params.npz"

        save_model_params(dummy_params, str(filepath))
        loaded_params = load_model_params(str(filepath))

        assert loaded_params.keys() == dummy_params.keys()

        for key in dummy_params:
            np.testing.assert_array_equal(
                loaded_params[key],
                dummy_params[key]
            )

    print("模型参数保存与加载测试通过！")

    print("\n" + "=" * 55)
    print("成员五补充测试全部通过！")
    print("=" * 55)


if __name__ == "__main__":
    main()
