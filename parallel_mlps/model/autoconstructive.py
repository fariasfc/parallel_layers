from copy import deepcopy
import logging
import math
from utils import helpers
from utils.accumulators import Accumulator, ObjectiveEnum
from typing import Any, List, Type
from torch.functional import Tensor
from model.parallel_mlp import ParallelMLPs, build_model_ids
from torch import nn
import torch
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


class AutoConstructive(nn.Module):
    def __init__(
        self,
        all_data_to_device: bool,
        loss_function: nn.Module,
        optimizer_cls: Type,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        num_workers: int,
        repetitions: int,
        activations: List[str],
        min_neurons: int,
        max_neurons: int,
        step_neurons: int,
        local_patience: int = 10,
        global_patience: int = 2,
        transform_data_strategy: str = "append_original_input",
        loss_rel_tol: float = 0.05,
        device: str = "cuda",
        logger: Any = None,
    ):
        super().__init__()
        self.all_data_to_device: bool = all_data_to_device
        self.loss_function: nn.Module = loss_function
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.repetitions = repetitions
        self.activations = activations
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.step_neurons = step_neurons
        self.local_patience = local_patience
        self.global_patience = global_patience
        self.transform_data_strategy = transform_data_strategy
        self.loss_rel_tol = loss_rel_tol
        self.device: str = device
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.hidden_neuron__model_id = build_model_ids(
            self.repetitions,
            self.activations,
            self.min_neurons,
            self.max_neurons,
            self.step_neurons,
        )

        self.best_model = None

    def _get_dataloader(self, x: Tensor, y: Tensor, shuffle=True):
        if self.all_data_to_device:
            x = x.to(self.device)
            y = y.to(self.device)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return dataloader

    def _get_best_mlp(
        self, train_dataloader: DataLoader, validation_dataloader: DataLoader
    ):
        loss_function = deepcopy(self.loss_function)
        loss_function.reduction = "none"

        in_features = train_dataloader.dataset[0][0].shape[0]
        out_features = len(train_dataloader.dataset.tensors[1].unique())

        pmlps = ParallelMLPs(
            in_features=in_features,
            out_features=out_features,
            hidden_neuron__model_id=self.hidden_neuron__model_id,
            activations=self.activations,
        )

        optimizer: Optimizer = self.optimizer_cls(
            params=pmlps.parameters(), lr=self.learning_rate
        )

        train_loss = Accumulator(
            name="train_loss",
            objective=ObjectiveEnum.MINIMIZATION,
            reduction_fn=lambda individual_losses: torch.mean(
                torch.cat(individual_losses, dim=0), dim=0
            ),
        )
        validation_loss = Accumulator(
            name="validation_loss",
            objective=ObjectiveEnum.MINIMIZATION,
            reduction_fn=lambda individual_losses: torch.mean(
                torch.cat(individual_losses, dim=0), dim=0
            ),
        )
        best_validation_loss = torch.tensor(float("inf"))
        current_patience = torch.zeros(pmlps.num_unique_models)

        t = tqdm(range(self.num_epochs))
        for epoch in t:
            self.train()
            reduced_train_loss = self.execute_loop(
                train_dataloader, loss_function, optimizer, pmlps, train_loss
            )  # [num_models]
            epoch_best_train_loss = reduced_train_loss.min().cpu().item()

            self.eval()
            with torch.no_grad():
                reduced_validation_loss = self.execute_loop(
                    validation_dataloader, loss_function, None, pmlps, validation_loss
                )  # [num_models]

            epoch_best_validation_loss = reduced_validation_loss.min().cpu().item()
            if epoch_best_validation_loss < best_validation_loss:
                best_validation_loss = epoch_best_validation_loss
                best_mlp = pmlps.extract_mlp(
                    helpers.min_ix_argmin(
                        reduced_validation_loss, pmlps.model_id__num_hidden_neurons
                    )
                )

            current_patience[~validation_loss.improved] += 1
            current_patience[validation_loss.improved] = 0

            model_ids_to_reset = torch.where(current_patience > self.local_patience)[0]

            if len(model_ids_to_reset) > 0:
                pmlps.reset_parameters(model_ids_to_reset)
                current_patience[model_ids_to_reset] = 0

            t.set_postfix(
                train_loss=epoch_best_train_loss,
                best_validation_loss=best_validation_loss,
            )

        return best_mlp, best_validation_loss

    def execute_loop(self, dataloader, loss_function, optimizer, pmlps, accumulator):
        accumulator.reset(reset_best=False)
        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            if optimizer:
                optimizer.zero_grad()

            outputs = pmlps(x)  # [batch_size, num_models, out_features]
            individual_losses = pmlps.calculate_loss(
                loss_func=loss_function, preds=outputs, target=y
            )
            accumulator.update(individual_losses)

            loss = individual_losses.mean(
                0
            ).sum()  # [batch_size, num_models] -> [num_models] -> []

            if optimizer:
                loss.backward()
                optimizer.step()
        reduced_accumulator = accumulator.apply_reduction()

        return reduced_accumulator

    def _transform_data(
        self,
        original_train_x,
        original_validation_x,
        current_train_x,
        current_validation_x,
        best_mlp,
    ):
        best_mlp.eval()
        with torch.no_grad():
            current_train_x = best_mlp(current_train_x)
            current_validation_x = best_mlp(current_validation_x)

            if self.transform_data_strategy == "append_original_input":
                current_train_x = torch.cat((original_train_x, current_train_x), dim=1)
                current_validation_x = torch.cat(
                    (original_validation_x, current_validation_x), dim=1
                )
        best_mlp.train()

        return current_train_x, current_validation_x

    def fit(
        self,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
    ):

        current_train_x = train_x
        current_train_y = train_y

        current_validation_x = validation_x
        current_validation_y = validation_y

        global_best_validation_loss = float("inf")

        current_model = nn.Sequential()

        current_patience = 0

        while current_patience < self.global_patience:
            train_dataloader = self._get_dataloader(current_train_x, current_train_y)
            validation_dataloader = self._get_dataloader(
                current_validation_x, current_validation_y
            )

            best_mlp, best_validation_loss = self._get_best_mlp(
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
            )

            current_model.add_module(name=f"{len(current_model)}", module=best_mlp)

            absolute_smaller_loss = best_validation_loss < global_best_validation_loss

            if absolute_smaller_loss:
                almost_same_loss = math.isclose(
                    best_validation_loss,
                    global_best_validation_loss,
                    rel_tol=self.loss_rel_tol,
                )

                if almost_same_loss:
                    improved_validation = False

                    self.logger.info(
                        f"best_validation_loss {best_validation_loss} is almost the same as global_best_validation_loss {global_best_validation_loss} considering rel_tol={self.loss_rel_tol}."
                    )
                else:
                    improved_validation = True
                    self.logger.info(
                        f"Improved validation loss from {global_best_validation_loss} to {best_validation_loss}. Added layer number {len(current_model)-1}. current_patience=0."
                    )
                    global_best_validation_loss = best_validation_loss
                    current_patience = 0
                    self.best_model = deepcopy(current_model)

            if not improved_validation:
                current_patience += 1
                self.logger.info(
                    f"No improvement. current_patience={current_patience}."
                )

            current_train_x, current_validation_x = self._transform_data(
                train_x, validation_x, current_train_x, current_validation_x, best_mlp
            )

    def predict(self, x: Tensor):
        return self.best_model(x)