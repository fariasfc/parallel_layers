from typing import Callable, Dict, List, Tuple
from torch import nn
import torch
from torch.functional import Tensor

from enum import Enum

from autoconstructive.utils import helpers


class ObjectiveEnum(Enum):
    MAXIMIZATION = "maximization"
    MINIMIZATION = "minimization"


class Objective:
    def __init__(
        self,
        name: str,
        objective: ObjectiveEnum = ObjectiveEnum.MINIMIZATION,
        reduction_fn: Callable[[Tensor], Tensor] = torch.mean,
        min_relative_improvement: float = 0.0,
    ) -> None:
        self.name = name
        self.objective = objective
        self.reduction_fn = reduction_fn
        self.min_relative_improvement = min_relative_improvement

        self.tensors = []
        self.current_reduction = None
        self.best = None
        self._improved = False
        self.is_dirty = True

    def set_values(self, values: Tensor):
        self.is_dirty = True
        self._improved = False
        self.tensors = values
        self.apply_reduction()

    def accumulate_values(self, values: Tensor):
        self.is_dirty = True
        self._improved = False
        self.tensors.append(values)

    def reset(self, reset_best=False):
        self.tensors = []
        self.current_reduction = None
        if reset_best:
            self.best = None
        self._improved = False
        self.is_dirty = True

    def get_initial_value(self):
        if self.objective == ObjectiveEnum.MAXIMIZATION:
            value = -float("inf")
        else:
            value = float("inf")
        return value

    def reset_best_from_ids(self, model_ids_to_reset):
        value = self.get_initial_value()

        if self.best is not None:
            self.best[model_ids_to_reset] = value

    def get_best_k_ids(self, best_k=1):
        best_ids_order = self.best.argsort()
        if self.objective == ObjectiveEnum.MAXIMIZATION:
            best_ids_order = best_ids_order.flip(0)

        return best_ids_order[:best_k]

    @property
    def improved(self):
        if self.is_dirty:
            raise RuntimeError("You should call reduce before checking if it improved.")

        return self._improved

    def apply_reduction(self):
        self.is_dirty = False

        if self.reduction_fn is not None:
            self.current_reduction = self.reduction_fn(self.tensors)
        else:
            self.current_reduction = self.tensors

        if self.best is None:
            self.best = self.current_reduction.clone()
            self._improved = torch.ones_like(self.current_reduction).bool()

        # elif self.objective == ObjectiveEnum.MAXIMIZATION:
        #     self._improved = self.current_reduction > self.best
        #     helpers.has_improved(self.current_reduction, self.best, self.min_improvement)

        # else:
        #     self._improved = self.current_reduction < self.best
        else:
            _, self._improved = helpers.has_improved(
                self.current_reduction,
                self.best,
                self.min_relative_improvement,
                self.objective,
            )

        if torch.any(self._improved):
            self.best[self._improved] = self.current_reduction[self._improved]

        return self.current_reduction


class MultiAccumulators:
    def __init__(self, names: List[Tuple]) -> None:
        # names = [(loss, ObjectiveEnum.MINIMIZATION)]
        self.accumulators = {
            name: Objective(name, objective) for (name, objective) in names
        }

    def update(self, dict_list_values: Dict[str, Tensor]):
        for k, v in dict_list_values.items():
            self.accumulators[k].append(v)

    def compute_average(self):
        [accumulator for accumulator in self.accumulators]
