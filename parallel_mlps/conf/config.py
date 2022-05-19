from ctypes import Union
from dataclasses import dataclass
from torch import optim
from typing import List, Optional
import hydra
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.nn.modules import loss
from torch.optim.optimizer import Optimizer

from parallel_mlps.autoconstructive.autoconstructive_enums import ObjectiveEnum


@dataclass
class TrainingConfig:
    dataset: str
    distance_name: Optional[str]
    data_home: str
    n_splits: int

    experiment_num: int

    project_name: str
    validation_rate_from_train: Optional[float]
    debug_test: bool
    reset_exhausted_models: bool

    num_epochs: int
    batch_size: int
    drop_samples: Optional[float]
    input_perturbation_strategy: Optional[str]
    regularization_gamma: Optional[float]
    monitored_metric: str
    monitored_metric_add_layers: str
    monitored_objective: ObjectiveEnum
    pareto_frontier: bool
    find_num_neurons_first: bool
    mcdm_weights: Optional[List[float]]


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
    topk_architecture: Optional[int]
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
    improvement_strategy: str
    device: str
    chosen_policy: str


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
    "Sigmoid()": nn.Sigmoid,
    "relu": nn.ReLU,
    "ReLU()": nn.ReLU,
    "tanh": nn.Tanh,
    "Tanh()": nn.Tanh,
    "selu": nn.SELU,
    "SELU()": nn.SELU,
    "leakyrelu": nn.LeakyReLU,
    "LeakyReLU(negative_slope=0.01)": nn.LeakyReLU,
    "identity": nn.Identity,
    "Identity()": nn.Identity,
    "elu": nn.ELU,
    "ELU()": nn.ELU,
    "gelu": nn.GELU,
    "GELU()": nn.GELU,
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
