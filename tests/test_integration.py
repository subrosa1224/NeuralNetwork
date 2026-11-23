import gzip
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from config import ExperimentConfig
from core.losses import build_loss
from experiment import split_training
from models.factory import build_model, prepare_inputs
from training import SGD, evaluate, fit
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.dataset import BatchLoader, TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS
from utils.predict import predict_and_show


@pytest.mark.parametrize("model_name", ["mlp", "cnn"])
@pytest.mark.parametrize("loss_name", ["cross_entropy", "mse"])
def test_models_learn_through_shared_training_interface(model_name, loss_name):
    np.random.seed(42)
    config = ExperimentConfig(model=model_name, hidden_sizes=[8], num_filters=2)
    images = np.zeros((16, 28, 28))
    labels = np.arange(16) % 2
    images[labels == 0, :14, :] = 1.0
    images[labels == 1, 14:, :] = 1.0
    images = prepare_inputs(images, model_name)
    model = build_model(config)
    loader = BatchLoader(images, labels, batch_size=8, shuffle=False)
    loss_fn = build_loss(loss_name)
    initial = evaluate(model, loader, loss_fn)["loss"]
    history = fit(model, loader, loader, loss_fn, epochs=4, optimizer=SGD(model, 0.03))
    assert history[-1]["val_loss"] < initial


@pytest.mark.parametrize("model_name", ["mlp", "cnn"])
def test_checkpoint_restores_structure_and_predictions(tmp_path, model_name):
    np.random.seed(7)
    config = ExperimentConfig(model=model_name, hidden_sizes=[11, 7], activation="tanh", num_filters=2, kernel_size=5, stride=2)
    model = build_model(config)
    images = prepare_inputs(np.random.random((3, 28, 28)), model_name)
    expected = model.forward(images)
    path = tmp_path / "nested" / "model.npz"
    save_checkpoint(model, config, path)
    restored, restored_config = load_checkpoint(path)
    assert restored_config.to_dict() == config.to_dict()
    np.testing.assert_array_equal(restored.forward(images), expected)


@pytest.mark.parametrize("model_name", ["mlp", "cnn"])
def test_single_sample_prediction_plot(tmp_path, model_name):
    config = ExperimentConfig(model=model_name, hidden_sizes=[4], num_filters=2)
    images = prepare_inputs(np.zeros((2, 28, 28)), model_name)
    path = tmp_path / "prediction.png"
    figure = predict_and_show(build_model(config).forward, images, np.array([0, 1]), num_samples=1, output_path=path, show=False, seed=42)
    assert figure.axes[0].images[0].get_array().shape == (28, 28)
    assert path.stat().st_size > 0


def test_training_validation_split_is_disjoint_and_repeatable():
    images = np.arange(20)
    config = ExperimentConfig(validation_size=5, train_limit=7)
    train, validation = split_training(images, images, config)
    repeated_train, repeated_validation = split_training(images, images, config)
    assert len(train[0]) == 7
    assert len(validation[0]) == 5
    assert not set(train[0]) & set(validation[0])
    np.testing.assert_array_equal(train[0], repeated_train[0])
    np.testing.assert_array_equal(validation[0], repeated_validation[0])


def _write_idx(data_dir, image_name, label_name, count):
    images = np.random.default_rng(count).integers(0, 256, size=(count, 28, 28), dtype=np.uint8)
    with gzip.open(data_dir / image_name, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, count, 28, 28))
        stream.write(images.tobytes())
    with gzip.open(data_dir / label_name, "wb") as stream:
        stream.write(struct.pack(">II", 2049, count))
        stream.write((np.arange(count) % 10).astype(np.uint8).tobytes())


def test_cli_train_evaluate_and_predict(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_idx(data, TRAIN_IMAGES, TRAIN_LABELS, 16)
    _write_idx(data, TEST_IMAGES, TEST_LABELS, 6)
    output = tmp_path / "run"
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, MPLBACKEND="Agg")

    def cli(*args):
        subprocess.run([sys.executable, "main.py", *args], cwd=project, env=environment, check=True, capture_output=True, text=True)

    cli("train", "--model", "cnn", "--num-filters", "2", "--epochs", "1", "--batch-size", "4", "--validation-size", "4", "--train-limit", "8", "--test-limit", "4", "--data-dir", str(data), "--output-dir", str(output))
    for name in ("config.json", "metrics.json", "history.csv", "model.npz", "curves.png", "predictions.png"):
        assert (output / name).stat().st_size > 0
    metrics = json.loads((output / "metrics.json").read_text())
    assert (metrics["train_samples"], metrics["validation_samples"], metrics["test_samples"]) == (8, 4, 4)
    report = tmp_path / "evaluation.json"
    cli("evaluate", "--checkpoint", str(output / "model.npz"), "--test-limit", "3", "--output", str(report))
    assert json.loads(report.read_text())["support"] == 3
    image = tmp_path / "single.png"
    cli("predict", "--checkpoint", str(output / "model.npz"), "--num-samples", "1", "--output", str(image))
    assert image.stat().st_size > 0
