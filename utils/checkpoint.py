"""把成员 5 的数组读写接口接到 MLP/CNN，并在文件中保存重建所需配置。"""

import json

import numpy as np

from config import ExperimentConfig
from models.cnn import CNN
from models.factory import build_model
from models.mlp import MLP
from utils.predict import load_model_params, save_model_params


def named_parameters(model):
    if isinstance(model, MLP):
        layers = [(f"layers.{index}", layer) for index, layer in enumerate(model.layers)]
    elif isinstance(model, CNN):
        layers = [("conv", model.conv), ("fc", model.fc)]
    else:
        raise TypeError(f"不支持的模型类型：{type(model).__name__}")
    for name, layer in layers:
        yield f"{name}.W", layer.W
        yield f"{name}.b", layer.b


def save_checkpoint(model, config, path):
    state = {name: parameter for name, parameter in named_parameters(model)}
    state["__config__"] = np.asarray(json.dumps(config.to_dict(), ensure_ascii=False))
    save_model_params(state, path)


def load_checkpoint(path):
    state = load_model_params(path)
    config = ExperimentConfig(**json.loads(str(state.pop("__config__").item())))
    model = build_model(config)
    parameters = dict(named_parameters(model))
    if state.keys() != parameters.keys():
        raise ValueError("检查点参数名称与模型结构不一致")
    for name, parameter in parameters.items():
        if state[name].shape != parameter.shape:
            raise ValueError(f"检查点参数 {name} 形状不一致：{state[name].shape} != {parameter.shape}")
        parameter[...] = state[name]
    return model, config
