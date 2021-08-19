from typing import Callable, Dict, List, Tuple
from torch import nn
import torch
from torch.functional import Tensor

from enum import Enum


class ObjectiveEnum(Enum):
    MAXIMIZATION = "maximization"
    MINIMIZATION = "minimization"


class Accumulator:
    def __init__(
        self,
        name: str,
        objective: ObjectiveEnum = ObjectiveEnum.MINIMIZATION,
        reduction_fn: Callable[[Tensor], Tensor] = torch.mean,
    ) -> None:
        self.name = name
        self.objective = objective
        self.reduction_fn = reduction_fn

        self.tensors = []
        self.current_reduction = None
        self.best = None
        self._improved = False
        self.is_dirty = True

    def update(self, values: Tensor):
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

    @property
    def improved(self):
        if self.is_dirty:
            raise RuntimeError("You should call reduce before checking if it improved.")

        return self._improved

    def apply_reduction(self):
        self.is_dirty = False

        self.current_reduction = self.reduction_fn(self.tensors)

        if self.best is None:
            self.best = self.current_reduction.clone()
            self._improved = torch.ones_like(self.current_reduction).bool()

        elif self.objective == ObjectiveEnum.MAXIMIZATION:
            self._improved = self.current_reduction > self.best

        else:
            self._improved = self.current_reduction < self.best

        if torch.any(self._improved):
            self.best[self._improved] = self.current_reduction[self._improved]

        return self.current_reduction


class MultiAccumulators:
    def __init__(self, names: List[Tuple]) -> None:
        # names = [(loss, ObjectiveEnum.MINIMIZATION)]
        self.accumulators = {
            name: Accumulator(name, objective) for (name, objective) in names
        }

    def update(self, dict_list_values: Dict[str, Tensor]):
        for k, v in dict_list_values.items():
            self.accumulators[k].append(v)

    def compute_average(self):
        [accumulator for accumulator in self.accumulators]
