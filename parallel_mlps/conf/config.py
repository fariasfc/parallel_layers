from dataclasses import dataclass
from torch import optim
from typing import List
import hydra
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.nn.modules import loss


@dataclass
class TrainingConfig:
    dataset: str
    data_home: str
    n_splits: int

    experiment_num: int

    project_name: str
    validation_rate_from_train: float


@dataclass
class ModelConfig:
    all_data_to_device: bool
    loss_function: str
    optimizer_cls: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    repetitions: int
    activations: List[str]
    min_neurons: int
    max_neurons: int
    step_neurons: int
    local_patience: int
    global_patience: int
    transform_data_strategy: str
    loss_rel_tol: float
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


def resolve_optimizer_type(optimizer_name):
    if optimizer_name == "adam":
        return optim.Adam
    #     return optim.Adam(**kwargs)
    else:
        ValueError(f"Optimizer {optimizer_name} not recognized.")


def resolve_activations(list_activations):
    act_list = []
    for act in list_activations:
        if act == "relu":
            act_list.append(nn.ReLU())

        elif act == "sigmoid":
            act_list.append(nn.Sigmoid())

        elif act == "tanh":
            act_list.append(nn.Tanh())
        elif act == "selu":
            act_list.append(nn.SELU())
        elif act == "leakyrelu":
            act_list.append(nn.LeakyReLU())
        elif act == "identity":
            act_list.append(nn.Identity())
        elif act == "elu":
            act_list.append(nn.ELU())
        elif act == "gelu":
            act_list.append(nn.GELU())
        else:
            raise ValueError(f"Activation {act} not recognized.")

    return act_list


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=AutoConstructiveConfig)
# cs.store(name="training", node=Training)
