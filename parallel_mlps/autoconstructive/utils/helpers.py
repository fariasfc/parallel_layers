import torch
import math
from torch.nn import init

from autoconstructive.utils.accumulators import ObjectiveEnum
from experiment_utils import assess_model


def has_improved(
    current_best_losses,
    global_best_losses,
    min_relative_improvement,
    objective_enum=ObjectiveEnum.MINIMIZATION,
    eps=1e-5,
    topk=1,
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
    percentage_of_best_loss = current_best_losses / (global_best_losses + eps)
    if objective_enum == ObjectiveEnum.MINIMIZATION:
        better_models_mask = percentage_of_best_loss < (1 - min_relative_improvement)
    else:
        better_models_mask = percentage_of_best_loss > (1 - min_relative_improvement)

    # current_best_losses.isinf()

    return percentage_of_best_loss, better_models_mask


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