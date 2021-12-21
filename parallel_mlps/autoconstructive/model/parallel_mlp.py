from copy import deepcopy
from conf.config import MAP_ACTIVATION
from functools import partial
from itertools import groupby
from autoconstructive.utils import helpers
from joblib import Parallel, delayed
import numpy as np
import math
from typing import Any, Counter, List, Optional
from torch import nn, random
from torch._C import Value
from torch.functional import Tensor
from torch.nn import init
import torch
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter
from torch.multiprocessing import Pool, set_start_method, freeze_support


class MLP(nn.Module):
    def __init__(self, hidden_layer, out_layer, activation, model_id, metadata, device):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.activation = activation
        self.model_id = model_id
        self.metadata = metadata
        self.device = device

    @property
    def out_features(self):
        return self.out_layer.out_features

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)

        return x


def build_model_ids(
    repetitions: int,
    activation_functions: list,
    min_neurons: int,
    max_neurons: int,
    step: int,
):
    """Creates a list with model ids to relate to hidden representations.
    1. Creates a list containing the number of hidden neurons for each architecture (independent of activation functions and/or repetitions)
    using the following formula neurons_structures=range(min_neurons, max_neurons+1, step)
    2. Calculates the number of independent models (parallel mlps) = len(neurons_structures) * len(activations) * repetitions

    Raises:
        ValueError: [description]
        ValueError: [description]
        RuntimeError: [description]
        ValueError: [description]

    Returns:
        hidden_neurons__model_id: List indicating for each global neuron the model_id that it belongs to
        output__model_id: List containing the id of the model for each output
        output__architecture_id: List containing the id of the architecture (neuron structure AND activation function) that the output belongs.
            Architectures with the same id means that it only differs the repetition number, but have equal neuron structure and activation function.
    """
    # Internal structure:
    # https://docs.google.com/spreadsheets/d/1ncnSjbDLGrNXkiT5Vo17DieIQJZZROM_znBsKlrwFnc/edit#gid=898813889
    if len(activation_functions) == 0:
        raise ValueError(
            "At least one activation function must be passed. Try `nn.Identity()` if you want no activation."
        )

    activation_names = [a.__class__.__name__ for a in activation_functions]
    if len(set(activation_names)) != len(activation_names):
        raise ValueError("activation_functions must have only unique values.")

    num_activations = len(activation_functions)

    neurons_structure = torch.arange(min_neurons, max_neurons + 1, step).tolist()
    num_different_neurons_structures = len(neurons_structure)
    num_different_architectures = num_different_neurons_structures * num_activations
    num_parallel_mlps = num_different_architectures * repetitions

    i = 0
    hidden_neuron__model_id = []
    while i < num_parallel_mlps:
        for structure in neurons_structure:
            hidden_neuron__model_id += [i] * structure
            i += 1

    # Mapping output to repetition. Output here means the number of models (model_id)
    output__repetition = []
    activation_ids = []
    for act_id in range(num_activations):
        activation_ids += [act_id] * (num_parallel_mlps // num_activations)
        for arch_id in range(num_different_architectures):
            output__repetition += [
                arch_id % num_different_architectures
            ] * num_different_architectures

    output__model_id = [i[0] for i in groupby(hidden_neuron__model_id)]
    output__architecture_id = (
        output__model_id[: num_activations * num_different_neurons_structures]
        * repetitions
    )

    return hidden_neuron__model_id, output__model_id, output__architecture_id


class ParallelMLPs(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons_structures: List[int],
        repetitions: int,
        drop_samples: float,
        input_perturbation_strategy: Optional[str],
        activations: List[nn.Module],
        bias: bool = True,
        device: str = "cuda",
        logger: Any = None,
    ):
        super().__init__()

        if len(activations) != len(set(activations)):
            raise ValueError(f"activations must contain only unique values.")

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations
        self.logger = logger
        self.drop_samples = drop_samples

        num_different_activations = len(activations)
        num_different_neurons_structures = len(neurons_structures)
        num_different_architectures = (
            num_different_neurons_structures * num_different_activations
        )
        num_total_models = num_different_architectures * repetitions

        num_models = 0
        model_id__activation_id = []
        model_id__num_hidden_neurons = []
        model_id__repetitions = []
        model_id__architecture_id = []
        hidden_neuron__model_id = []

        for act_id in range(num_different_activations):
            model_id__activation_id += [act_id] * (
                num_different_neurons_structures * repetitions
            )

            for rep_id in range(repetitions):
                model_id__repetitions += [rep_id] * num_different_neurons_structures

                model_id__architecture_id += list(
                    np.arange(num_different_neurons_structures)
                    + act_id * num_different_neurons_structures
                )

                model_id__num_hidden_neurons += neurons_structures

                for neurons_structure in neurons_structures:
                    hidden_neuron__model_id += [num_models] * neurons_structure
                    num_models += 1

        if num_models != len(set(hidden_neuron__model_id)):
            raise RuntimeError(f"num_models != model_id__num_hidden_neurons!")

        self.model_id = torch.Tensor(range(num_models)).long().to(self.device)

        # Mappings: index -> id
        self.hidden_neuron__model_id = (
            torch.Tensor(hidden_neuron__model_id).long().to(self.device)
        )

        # self.output__model_id = torch.Tensor(output__model_id).long().to(self.device)
        self.model_id__architecture_id = (
            torch.Tensor(model_id__architecture_id).long().to(self.device)
        )

        self.total_num_hidden_neurons = len(self.hidden_neuron__model_id)
        self.unique_model_ids = sorted(list(set(hidden_neuron__model_id)))

        if len(self.unique_model_ids) != len(self.model_id):
            raise RuntimeError(f"unique_model_ids != model_id")

        self.model_id__num_hidden_neurons = torch.from_numpy(
            np.bincount(self.hidden_neuron__model_id.cpu().numpy())
        ).to(self.device)

        if np.any(
            model_id__num_hidden_neurons
            != self.model_id__num_hidden_neurons.cpu().numpy()
        ):
            raise RuntimeError(
                f"modeL_id__num_hidden_neurons != self.model_id__num_hidden_neurons"
            )

        # self.model_id__start_neuron = torch.zeros_like(self.model_id__num_hidden_neurons)
        self.model_id__start_idx = torch.cat(
            [
                torch.tensor([0]).to(self.device),
                self.model_id__num_hidden_neurons.cumsum(0)[:-1],
            ]
        )
        self.model_id__end_idx = (
            self.model_id__start_idx + self.model_id__num_hidden_neurons
        )
        # self.model_id__start_neuron =

        # self.model_id__num_hidden_neurons = torch.bincount(
        #     self.hidden_neuron__model_id
        # ).to(self.device)

        self.num_unique_models = len(self.unique_model_ids)
        self.num_activations = len(activations)

        self.activations_split = self.total_num_hidden_neurons // self.num_activations

        self.shared_hidden_layer = nn.Linear(
            self.in_features, self.total_num_hidden_neurons
        )
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.total_num_hidden_neurons)
        )
        if bias:
            self.bias = Parameter(
                torch.Tensor(self.num_unique_models, self.out_features)
            )
        else:
            self.bias = None
            self.register_parameter("bias", None)

        self.input_perturbation_strategy = input_perturbation_strategy

        self.input_perturbation = (
            nn.Parameter(
                torch.ones_like(self.shared_hidden_layer.weight),
                requires_grad=False,
            )
            if self.input_perturbation_strategy is not None
            else None
        )

        self.reset_parameters()
        self.enforce_input_perturbation()
        self.to(device)
        self.logger.info(f"Model sent to {device}!")

    def _init_input_perturbation(self, input_perturbation):
        r = None
        if input_perturbation == "sqrt":
            input_perturbation_threshold = (
                int(np.sqrt(self.in_features)) / self.in_features
            )
            r = nn.Parameter(
                torch.rand_like(self.shared_hidden_layer.weight)
                < input_perturbation_threshold,
                requires_grad=False,
            )
        return r

    def enforce_input_perturbation(self):
        if self.input_perturbation is not None:
            with torch.no_grad():
                self.shared_hidden_layer.weight *= self.input_perturbation

    def _build_outputs_ids(self):
        return [i[0] for i in groupby(self.hidden_neuron__model_id)]

    def reset_parameters(self, layer_ids=None):
        if layer_ids == None:
            layer_ids = self.unique_model_ids

        with torch.no_grad():
            for layer_id in layer_ids:
                start = self.model_id__start_idx[layer_id]
                end = self.model_id__end_idx[layer_id]
                hidden_w = self.shared_hidden_layer.weight[start:end, :]
                hidden_b = self.shared_hidden_layer.bias[start:end]

                out_w = self.weight[:, start:end]
                out_b = self.bias[layer_id, :]

                for w, b in [(hidden_w, hidden_b), (out_w, out_b)]:
                    init.kaiming_uniform_(w, a=math.sqrt(5))
                    fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(b, -bound, bound)

                if self.input_perturbation_strategy is not None:
                    self.reset_input_perturbation(start, end)

    def reset_input_perturbation(self, start, end):
        hidden_w = self.shared_hidden_layer.weight[start:end, :]
        input_perturbation = self.input_perturbation[start:end, :]
        input_perturbation[:, :] = 1
        if "random_sparsity" in self.input_perturbation_strategy:
            sparsity = float(
                self.input_perturbation_strategy.replace("random_sparsity", "")
            )
            sparse_mask = torch.rand_like(hidden_w) < sparsity
            hidden_w[:, :] = hidden_w[:, :] * sparse_mask
            input_perturbation[:, :] = sparse_mask

            return
        elif self.input_perturbation_strategy == "sqrt":
            num_used_features = max(1, int(np.sqrt(self.in_features)))
        elif self.input_perturbation_strategy == "fromsqrt":
            sqrt_value = max(1, int(np.sqrt(self.in_features)))
            num_used_features = max(sqrt_value, int(torch.rand(1) * self.in_features))

        if self.input_perturbation_strategy is not None:
            not_used_features = torch.randperm(self.in_features)[num_used_features:]

            hidden_w[:, not_used_features] = 0
            self.input_perturbation[start:end, :] = 1
            self.input_perturbation[start:end, not_used_features] = 0

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
        x = self.shared_hidden_layer(x)  # [batch_size, total_hidden_neurons]
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

    def extract_mlps(self, model_ids: List[int]) -> List[MLP]:
        """Extracts a completely independent MLP."""
        if max(model_ids) >= self.num_unique_models:
            raise ValueError(
                f"model_id {max(model_ids)} > num_uniqe_models {self.num_unique_models}"
            )

        mlps = []
        with torch.no_grad():
            for model_id in model_ids:
                model_neurons = self.hidden_neuron__model_id == model_id
                hidden_weight = self.shared_hidden_layer.weight[model_neurons, :]
                hidden_bias = self.shared_hidden_layer.bias[model_neurons]

                out_weight = self.weight[:, model_neurons]
                out_bias = self.bias[model_id, :]

                hidden_layer = nn.Linear(
                    in_features=hidden_weight.shape[1],
                    out_features=hidden_weight.shape[0],
                )
                activation = self.get_activation_from_model_id(model_id)
                out_layer = nn.Linear(
                    in_features=hidden_layer.out_features,
                    out_features=self.out_features,
                )

                hidden_layer.weight[:, :] = hidden_weight.clone()
                hidden_layer.bias[:] = hidden_bias.clone()

                out_layer.weight[:, :] = out_weight.clone()
                out_layer.bias[:] = out_bias.clone()

                mlps.append(
                    MLP(
                        hidden_layer=hidden_layer,
                        out_layer=out_layer,
                        activation=activation,
                        model_id=model_id,
                        metadata={},
                        device=self.device,
                    ).to(self.device)
                )

        return mlps

    def get_model_ids_from_architecture_id(self, architecture_id):
        indexes = self.model_id__architecture_id == architecture_id
        model_ids = self.model_id[indexes]
        return model_ids.cpu().tolist()

    def get_num_hidden_neurons_from_architecture_id(self, architecture_id):
        model_id = self.get_model_ids_from_architecture_id(architecture_id)[0]
        return self.get_num_hidden_neurons_from_model_id(model_id)

    def get_num_hidden_neurons_from_model_id(self, model_id):
        return ((self.hidden_neuron__model_id == model_id)).sum()

    def get_architecture_id_from_model_id(self, model_id):
        index = self.model_id == model_id
        architecture_id = self.model_id__architecture_id[index][0]
        return architecture_id

    def get_activation_from_model_id(self, model_id):
        activation_index = (
            torch.nonzero(self.hidden_neuron__model_id == model_id)[0]
            // self.activations_split
        )
        activation = deepcopy(self.activations[activation_index])
        return activation

    def get_activation_name_from_model_id(self, model_id):
        activation = self.get_activation_from_model_id(model_id)
        activation_name = [
            k for (k, v) in MAP_ACTIVATION.items() if v == type(activation)
        ][0]
        return activation_name

    # def get_regularization_term(self) -> torch.Tensor:
    #     self.hidden_layer.weight
    #     torch.zeros(
    #         self.num_unique_models, self.out_features, device=self.device
    #     ).scatter_add_()
    #     # [batch_size, total_repetitions, num_architectures, out_features]
    #     adjusted_out = (
    #         torch.zeros(
    #             batch_size, self.num_unique_models, self.out_features, device=x.device
    #         ).scatter_add_(
    #             1,
    #             # self.hidden_neuron__layer_id,
    #             self.hidden_neuron__model_id[None, :, None].expand(
    #                 batch_size, -1, self.out_features
    #             ),
    #             x,
    #         )
    #     ) + self.bias[None, :, :]

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
