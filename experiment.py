"""统一组织数据划分、模型训练、评估、检查点与展示。"""

import csv
import json

import numpy as np

from core.losses import build_loss
from models.factory import build_model, prepare_inputs
from training import SGD, evaluate, fit
from utils.checkpoint import save_checkpoint
from utils.dataset import BatchLoader, get_mnist_data
from utils.predict import predict_and_show
from utils.visualization import plot_history


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_training(images, labels, config):
    if config.validation_size >= len(labels):
        raise ValueError("validation_size 必须小于训练集样本数")
    indices = np.random.default_rng(config.seed).permutation(len(labels))
    validation = indices[:config.validation_size]
    train = indices[config.validation_size:][:config.train_limit]
    return (images[train], labels[train]), (images[validation], labels[validation])


def run_training(config, show=False):
    np.random.seed(config.seed)
    (images, labels), (test_images, test_labels) = get_mnist_data(config.data_dir, flatten=False)
    (train_images, train_labels), (validation_images, validation_labels) = split_training(images, labels, config)
    test_images, test_labels = test_images[:config.test_limit], test_labels[:config.test_limit]
    train_images = prepare_inputs(train_images, config.model)
    validation_images = prepare_inputs(validation_images, config.model)
    test_images = prepare_inputs(test_images, config.model)

    print(f"model={config.model}  train={len(train_labels)}  validation={len(validation_labels)}  test={len(test_labels)}", flush=True)
    model = build_model(config)
    loss_fn = build_loss(config.loss)
    history = fit(
        model, BatchLoader(train_images, train_labels, config.batch_size),
        BatchLoader(validation_images, validation_labels, config.batch_size, shuffle=False),
        loss_fn, config.epochs, SGD(model, config.learning_rate),
    )
    test_metrics = evaluate(model, BatchLoader(test_images, test_labels, config.batch_size, shuffle=False), loss_fn)
    result = {
        "model": config.model, "seed": config.seed,
        "train_samples": len(train_labels), "validation_samples": len(validation_labels),
        "test_samples": len(test_labels), "test": test_metrics,
    }
    output = config.run_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "metrics.json", result)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    save_checkpoint(model, config, output / "model.npz")
    plot_history(history, output / "curves.png", show=show)
    predict_and_show(
        model.forward, test_images, test_labels, num_samples=min(10, len(test_labels)),
        output_path=output / "predictions.png", show=show, seed=config.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"实验结果已保存至 {output.resolve()}")
    return result
