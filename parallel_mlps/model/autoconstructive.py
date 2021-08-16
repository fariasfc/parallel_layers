from copy import deepcopy
from utils.accumulators import Accumulator, ObjectiveEnum
from typing import List, Type
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
        device: str = "cuda",
    ):
        super().__init__()
        self.all_data_to_device: bool = all_data_to_device
        self.device: str = device
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
        self.hidden_neuron__model_id = build_model_ids(
            self.repetitions,
            self.activations,
            self.min_neurons,
            self.max_neurons,
            self.step_neurons,
        )

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

    def _get_mlp(self, train_dataloader: DataLoader, validation_dataloader: DataLoader):
        loss_function = deepcopy(self.loss_function)
        loss_function.reduction = "none"

        in_features = train_dataloader.dataset[0][0].shape
        out_features = len(train_dataloader.dataset[0][1])

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
                torch.stack(individual_losses), dim=0
            ),
        )
        validation_loss = Accumulator(
            name="validation_loss",
            objective=ObjectiveEnum.MINIMIZATION,
            reduction_fn=lambda individual_losses: torch.mean(
                torch.stack(individual_losses), dim=0
            ),
        )
        best_validation_loss = None

        t = tqdm(range(self.num_epochs))
        for epoch in t:
            self.train()
            reduced_train_loss = self.execute_loop(
                train_dataloader, loss_function, optimizer, pmlps, train_loss
            )

            self.eval()
            with torch.no_grad():
                reduced_validation_loss = self.execute_loop(
                    validation_dataloader, loss_function, None, pmlps, validation_loss
                )

            if validation_loss.improved:
                best_validation_loss = reduced_validation_loss

            t.set_postfix(
                {
                    f"train_loss: {reduced_train_loss.item()}, best_validation_loss: {best_validation_loss.item()}"
                }
            )

    def execute_loop(self, dataloader, loss_function, optimizer, pmlps, accumulator):
        accumulator.reset(reset_best=False)
        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            if optimizer:
                optimizer.zero_grad()

            outputs = pmlps(x)
            individual_losses = pmlps.calculate_loss(
                loss_func=loss_function, preds=outputs, target=y
            )
            accumulator.update(individual_losses)

            loss = individual_losses.mean(0)

            if optimizer:
                loss.backward()
                optimizer.step()
        reduced_accumulator = accumulator.apply_reduction()

        return reduced_accumulator

    def fit(
        self,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
    ):
        train_dataloader = self._get_dataloader(train_x, train_y)
        validation_dataloader = self._get_dataloader(validation_x, validation_y)

        self._get_mlp(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
        )
