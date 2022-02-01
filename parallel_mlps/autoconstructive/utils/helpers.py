import torch
import numpy as np
import math
from torch.nn import init

from autoconstructive.utils.accumulators import ObjectiveEnum
from experiment_utils import assess_model


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array of costs to be minimized
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def has_improved(
    current_epoch_best_metric,
    global_best_metric,
    min_relative_improvement,
    objective_enum=ObjectiveEnum.MINIMIZATION,
    eps=1e-5,
):
    """Verify if the current_best_losses is better than global_best_losses given a min_relative_improvement.

    Args:
        current_best_losses ([type]): [description]
        global_best_losses ([type]): [description]
        min_relative_improvement ([type]): [description]
        objective_enum ([type], optional): [description]. Defaults to ObjectiveEnum.MINIMIZATION.
        eps ([type], optional): [description]. Defaults to 1e-5.

    Returns:
        [type]: [description]
    """
    if isinstance(global_best_metric, float):
        global_best_metric = (
            torch.tensor([global_best_metric])
            .type(current_epoch_best_metric.dtype)
            .to(current_epoch_best_metric.device)
        )

    percentage_of_best_loss = current_epoch_best_metric / (global_best_metric + eps)

    if torch.all(global_best_metric.isinf()):
        better_models_mask = torch.ones_like(global_best_metric).bool()
    else:
        if objective_enum == ObjectiveEnum.MINIMIZATION:
            better_models_mask = percentage_of_best_loss < (
                1 - min_relative_improvement
            )
        else:
            better_models_mask = percentage_of_best_loss > (
                1 + min_relative_improvement
            )

    return percentage_of_best_loss, better_models_mask


def select_top_k(monitored_metric, better_models_mask, best_k):
    objective_enum = monitored_metric.objective
    best_metrics = monitored_metric.current_reduction
    if any(better_models_mask):
        better_models_ids = torch.nonzero(better_models_mask)
        best_k = min(best_k, better_models_ids.nelement())

        if objective_enum == ObjectiveEnum.MINIMIZATION:
            # topk return largest elements first by default
            topk_indices = torch.topk(
                best_metrics[better_models_mask], best_k, largest=False
            ).indices
        else:
            topk_indices = torch.topk(
                best_metrics[better_models_mask], best_k, largest=True
            ).indices

        better_models_ids = better_models_ids[topk_indices]
        better_models_mask = torch.zeros_like(better_models_mask).bool()
        better_models_mask[better_models_ids] = True

    return better_models_mask


def debug_assess_model(logits, y_labels, model_id):
    return assess_model(
        logits=logits[:, model_id, :].cpu().numpy(),
        y_labels=y_labels.cpu().numpy(),
        metric_prefix="",
    )


def min_ix_argmin(a, n_hidden, ignore_zeros=False, rtol=0):
    """Get the min value of a with the lowest n_hidden in case of draw.

    Args:
        a ([type]): [description]
        n_hidden ([type]): [description]
        ptol (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if rtol > 0:
        # Considering values in a as the same when compared to the max value given a tolerance.
        # old_a = a.clone()
        a = a.clone()
        min_a = torch.min(a)
        ixs = (min_a / a) >= (1 - rtol)
        a[ixs] = min_a

    if ignore_zeros:
        # min_value = np.nanmin(a[a != 0])
        min_value = a[a != 0].min()
    else:
        # min_value = np.nanmin(a)
        min_value = a.min()
    min_ixs = torch.where(a == min_value)[0]
    min_hidden = torch.argmin(n_hidden[min_ixs])
    i = torch.min(min_hidden)
    ix = min_ixs[i]

    return ix


def _init_weights(w, b):
    init.kaiming_uniform_(w, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(w)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(b, -bound, bound)
    return w, b


def reset_parameters_model(
    hidden_layer, hidden_neuron__model_id, weight, bias, layer_id: int
):
    hidden_mask = hidden_neuron__model_id == layer_id
    hidden_w = hidden_layer.weight[hidden_mask, :]
    hidden_b = hidden_layer.bias[hidden_mask]

    out_w = weight[:, hidden_mask]
    out_b = bias[layer_id, :]

    hidden_w, hidden_b = _init_weights(hidden_w, hidden_b)
    hidden_layer.weight[hidden_mask, :] = hidden_w
    hidden_layer.bias[hidden_mask] = hidden_b

    out_w, out_b = _init_weights(out_w, out_b)
    weight[:, hidden_mask] = out_w
    bias[layer_id, :] = out_b