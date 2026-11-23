"""用 train / evaluate / predict 运行统一 MNIST 实验。"""

import argparse
import json
from pathlib import Path

from config import load_config
from core.losses import build_loss
from experiment import run_training, write_json
from models.factory import prepare_inputs
from training import evaluate
from utils.checkpoint import load_checkpoint
from utils.dataset import BatchLoader, get_mnist_data
from utils.predict import predict_and_show


def train_command(args):
    names = (
        "model", "hidden_sizes", "activation", "num_filters", "kernel_size", "stride", "padding",
        "loss", "learning_rate", "epochs", "batch_size", "seed", "validation_size",
        "train_limit", "test_limit", "data_dir", "output_dir",
    )
    config = load_config(args.config, **{name: getattr(args, name) for name in names})
    run_training(config, show=args.show)


def test_data(args, config):
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir
    _, (images, labels) = get_mnist_data(data_dir, flatten=False)
    if args.test_limit is not None and args.test_limit <= 0:
        raise ValueError("test_limit 必须大于 0")
    return prepare_inputs(images[:args.test_limit], config.model), labels[:args.test_limit]


def evaluate_command(args):
    model, config = load_checkpoint(args.checkpoint)
    images, labels = test_data(args, config)
    metrics = evaluate(model, BatchLoader(images, labels, config.batch_size, shuffle=False), build_loss(config.loss))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(Path(args.output), metrics)


def predict_command(args):
    model, config = load_checkpoint(args.checkpoint)
    images, labels = test_data(args, config)
    predict_and_show(
        model.forward, images, labels, num_samples=args.num_samples,
        output_path=args.output, show=args.show, seed=config.seed,
    )
    print(f"预测图片已保存至 {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description="NumPy MNIST 神经网络实验")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="训练模型并保存指标、曲线、权重和预测图片")
    train.add_argument("--config", help="JSON 配置文件")
    train.add_argument("--model", choices=("mlp", "cnn"))
    train.add_argument("--hidden-sizes", type=int, nargs="+")
    train.add_argument("--activation", choices=("relu", "sigmoid", "tanh"))
    train.add_argument("--loss", choices=("cross_entropy", "mse"))
    train.add_argument("--learning-rate", type=float)
    for name in ("num-filters", "kernel-size", "stride", "padding", "epochs", "batch-size", "seed", "validation-size", "train-limit", "test-limit"):
        train.add_argument(f"--{name}", type=int)
    train.add_argument("--data-dir")
    train.add_argument("--output-dir")
    train.add_argument("--show", action="store_true", help="同时打开 Matplotlib 图片窗口")
    train.set_defaults(func=train_command)

    for name, handler in (("evaluate", evaluate_command), ("predict", predict_command)):
        command = commands.add_parser(name, help="从保存的模型重建并评估" if name == "evaluate" else "从保存的模型预测并输出图片")
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--data-dir")
        command.add_argument("--test-limit", type=int)
        if name == "evaluate":
            command.add_argument("--output", help="可选：保存指标 JSON 的路径")
        else:
            command.add_argument("--output", default="results/predictions.png")
            command.add_argument("--num-samples", type=int, default=5)
            command.add_argument("--show", action="store_true")
        command.set_defaults(func=handler)
    return parser


def main():
    args = build_parser().parse_args()
    if not getattr(args, "show", False):
        import matplotlib
        matplotlib.use("Agg")
    args.func(args)


if __name__ == "__main__":
    main()
