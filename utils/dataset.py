import os
import gzip
import numpy as np
import urllib.request

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

def download_mnist(data_dir="data"):
    """自动下载 MNIST 数据集到指定 data 目录"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    files = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]
    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            print(f"正在下载数据集文件: {file} ...")
            urllib.request.urlretrieve(MNIST_URL + file, filepath)
    print("MNIST 数据集检查/下载完成！")

def _load_images(filepath):
    """使用 NumPy 解析 IDX 格式的图片二进制文件"""
    with gzip.open(filepath, 'r') as f:
        f.read(4) 
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows, cols)
    return data

def _load_labels(filepath):
    """使用 NumPy 解析 IDX 格式的标签二进制文件"""
    with gzip.open(filepath, 'r') as f:
        f.read(4)
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_mnist_data(data_dir="data", flatten=True, normalize=True):
    """
    对外提供的主函数：加载并处理 MNIST 数据集（满足划分与归一化要求）
    """
    download_mnist(data_dir)
    
    X_train = _load_images(os.path.join(data_dir, TRAIN_IMAGES))
    y_train = _load_labels(os.path.join(data_dir, TRAIN_LABELS))
    X_test = _load_images(os.path.join(data_dir, TEST_IMAGES))
    y_test = _load_labels(os.path.join(data_dir, TEST_LABELS))

    # 数据归一化 
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # 展平处理
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return (X_train, y_train), (X_test, y_test)

class BatchLoader:
    """实现 mini-batch 数据加载迭代器，供训练时循环调用"""
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        self.pointer = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.pointer >= self.num_samples:
            raise StopIteration
        
        batch_idx = self.indices[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self.X[batch_idx], self.y[batch_idx]