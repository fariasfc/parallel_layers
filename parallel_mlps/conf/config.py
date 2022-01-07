from dataclasses import dataclass
from torch import optim
from typing import List, Optional
import hydra
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.nn.modules import loss
from torch.optim.optimizer import Optimizer


@dataclass
class TrainingConfig:
    dataset: str
    distance_name: str
    data_home: str
    n_splits: int

    experiment_num: int

    project_name: str
    validation_rate_from_train: float
    debug_test: bool

    num_epochs: int
    batch_size: int
    drop_samples: float
    input_perturbation_strategy: str


@dataclass
class ModelConfig:
    all_data_to_device: bool
    loss_function: str
    optimizer_name: str
    learning_rate: float
    num_workers: int
    repetitions: int
    repetitions_for_best_neuron: int
    activations: List[str]
    topk: Optional[int]
    output_confidence: bool
    min_confidence: float
    min_neurons: int
    max_neurons: int
    max_layers: Optional[int]
    stack_hidden_layers: bool
    step_neurons: int
    local_patience: int
    global_patience: int
    transform_data_strategy: str
    strategy_select_best: str
    loss_rel_tol: float
    min_improvement: float
    device: str


@dataclass
class AutoConstructiveConfig:
    training: TrainingConfig
    model: ModelConfig


def resolve_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {loss_name} not recognized.")


def resolve_optimizer_type(cfg):
    optimizer_name = cfg.model.optimizer_cls
    if optimizer_name == "adam":
        return optim.Adam

    elif optimizer_name == "sgd":
        return optim.SGD
    #     return optim.Adam(**kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")


def create_optimizer(optimizer_name, learning_rate, parameters) -> Optimizer:
    if optimizer_name == "adam":
        optimizer = optim.Adam(params=parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            params=parameters, lr=learning_rate
        )  # , momentum=0.9, nesterov=True)
    #     return optim.Adam(**kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

    return optimizer


MAP_ACTIVATION = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "leakyrelu": nn.LeakyReLU,
    "identity": nn.Identity,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def resolve_activations(list_activations):
    act_list = []
    for act in list_activations:
        if act in MAP_ACTIVATION:
            act_list.append(MAP_ACTIVATION[act]())
        else:
            raise ValueError(f"Activation {act} not recognized.")

    return act_list


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=AutoConstructiveConfig)
# cs.store(name="training", node=Training)
