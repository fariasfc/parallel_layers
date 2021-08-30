from copy import deepcopy
from functools import partial
from autoconstructive.utils import helpers
from joblib import Parallel, delayed
import numpy as np
import math
from typing import Any, Counter, List
from torch import nn
from torch._C import Value
from torch.functional import Tensor
from torch.nn import init
import torch
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter
from torch.multiprocessing import Pool, set_start_method, freeze_support



def build_model_ids(
    repetitions: int,
    activation_functions: list,
    min_neurons: int,
    max_neurons: int,
    step: int,
):

    if len(activation_functions) == 0:
        raise ValueError(
            "At least one activation function must be passed. Try `nn.Identity()` if you want no activation."
        )

    activation_names = [a.__class__.__name__ for a in activation_functions]
    if len(set(activation_names)) != len(activation_names):
        raise ValueError("activation_functions must have only unique values.")

    num_activations = len(activation_functions)

    neurons_structure = torch.arange(min_neurons, max_neurons + 1, step).tolist()
    num_parallel_mlps = len(neurons_structure) * num_activations * repetitions

    i = 0
    layer_ids = []
    while i < num_parallel_mlps:
        for structure in neurons_structure:
            layer_ids += [i] * structure
            i += 1

    return layer_ids


class ParallelMLPs(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_neuron__model_id: List[int],
        activations: List[nn.Module],
        bias: bool = True,
        device: str = "cuda",
        logger: Any= None
    ):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations
        self.logger = logger

        self.hidden_neuron__model_id = (
            torch.Tensor(hidden_neuron__model_id).long().to(self.device)
        )
        self.total_hidden_neurons = len(self.hidden_neuron__model_id)
        self.unique_model_ids = sorted(list(set(hidden_neuron__model_id)))
        self.model_id__num_hidden_neurons = torch.from_numpy(
            np.bincount(self.hidden_neuron__model_id.cpu().numpy())
        ).to(self.device)

        # self.model_id__num_hidden_neurons = torch.bincount(
        #     self.hidden_neuron__model_id
        # ).to(self.device)

        self.num_unique_models = len(self.unique_model_ids)
        self.num_activations = len(activations)

        self.activations_split = self.total_hidden_neurons // self.num_activations

        self.hidden_layer = nn.Linear(self.in_features, self.total_hidden_neurons)
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.total_hidden_neurons)
        )
        if bias:
            self.bias = Parameter(
                torch.Tensor(self.num_unique_models, self.out_features)
            )
        else:
            self.bias = None
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.to(device)
        self.logger.info(f"Model sent to {device}!")



    def reset_parameters(self, layer_ids=None):
        if layer_ids == None:
            layer_ids = self.unique_model_ids

        start = 0
        with torch.no_grad():
            for layer_id, nb_hidden in enumerate(self.model_id__num_hidden_neurons):
                end = start + nb_hidden
                hidden_w = self.hidden_layer.weight[start:end, :]
                hidden_b = self.hidden_layer.bias[start:end]

                out_w = self.weight[:, start:end]
                out_b = self.bias[layer_id, :]

                for w, b in [(hidden_w, hidden_b), (out_w, out_b)]:
                    init.kaiming_uniform_(w, a=math.sqrt(5))
                    fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(b, -bound, bound)
                start = end

    def apply_activations(self, x: Tensor) -> Tensor:
        tensors = x.split(self.activations_split, dim=1)
        output = []
        sub_tensor_out_features = tensors[0].shape[1]
        for (act, sub_tensor) in zip(self.activations, tensors):
            if sub_tensor.shape[1] != sub_tensor_out_features:
                raise RuntimeError(
                    f"sub_tensors with different number of parameters per activation {[t.shape for t in tensors]}"
                )
            output.append(act(sub_tensor))
        output = torch.cat(output, dim=1)
        return output

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = self.hidden_layer(x)  # [batch_size, total_hidden_neurons]
        x = self.apply_activations(x)  # [batch_size, total_hidden_neurons]

        x = (
            x[:, :, None] * self.weight.T[None, :, :]
        )  # [batch_size, total_hidden_neurons, out_features]

        # [batch_size, total_repetitions, num_architectures, out_features]
        adjusted_out = (
            torch.zeros(
                batch_size, self.num_unique_models, self.out_features, device=x.device
            ).scatter_add_(
                1,
                # self.hidden_neuron__layer_id,
                self.hidden_neuron__model_id[None, :, None].expand(
                    batch_size, -1, self.out_features
                ),
                x,
            )
        ) + self.bias[None, :, :]

        # [batch_size, num_unique_models, out_features]
        return adjusted_out

    def calculate_loss(self, loss_func, preds, target):
        if hasattr(loss_func, "reduction"):
            assert loss_func.reduction == "none"

        if preds.ndim == 3:
            batch_size, num_models, neurons = preds.shape
            loss = loss_func(
                preds.permute(0, 2, 1), target[:, None].expand(-1, num_models)
            )
        else:
            loss = loss_func(preds, target)

        return loss

    def extract_mlp(self, model_id: int) -> nn.Sequential:
        if model_id >= self.num_unique_models:
            raise ValueError(
                f"model_id {model_id} > num_uniqe_models {self.num_unique_models}"
            )

        with torch.no_grad():
            model_neurons = self.hidden_neuron__model_id == model_id
            hidden_weight = self.hidden_layer.weight[model_neurons, :]
            hidden_bias = self.hidden_layer.bias[model_neurons]

            out_weight = self.weight[:, model_neurons]
            out_bias = self.bias[model_id, :]

            hidden_layer = nn.Linear(
                in_features=hidden_weight.shape[1], out_features=hidden_weight.shape[0]
            )
            activation_index = (
                torch.nonzero(self.hidden_neuron__model_id == model_id)[0]
                // self.activations_split
            )
            activation = deepcopy(self.activations[activation_index])
            out_layer = nn.Linear(
                in_features=hidden_layer.out_features, out_features=self.out_features
            )

            hidden_layer.weight[:, :] = hidden_weight.clone()
            hidden_layer.bias[:] = hidden_bias.clone()

            out_layer.weight[:, :] = out_weight.clone()
            out_layer.bias[:] = out_bias.clone()

        return nn.Sequential(hidden_layer, activation, out_layer).to(self.device)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
