from copy import deepcopy
import pandas as pd
from enum import Enum
# import pyinstrument

from conf import config
from time import perf_counter
import wandb
import numpy as np
import logging
import math
from autoconstructive.utils import helpers
from autoconstructive.utils.accumulators import Accumulator, ObjectiveEnum
from typing import Any, List, Type
from torch.functional import Tensor
from autoconstructive.model.parallel_mlp import ParallelMLPs, build_model_ids
from torch import nn
import torch
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

from conf.config import create_optimizer

class StrategySelectBestEnum(str, Enum):
    GLOBAL_BEST= 'global_best'
    ARCHITECTURE_MEDIAN_BEST= 'architecture_median_best' #select architecture: groupby(architecture).median => Get best model among selected architecture
    ARCHITECTURE_MEDIAN_MEDIAN= 'architecture_median_median' #select architecture: groupby(architecture).median => Get the median model among selected architecture

class AutoConstructiveModel(nn.Module):
    def __init__(
        self,
        all_data_to_device: bool,
        loss_function: nn.Module,
        optimizer_name: str,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        num_workers: int,
        repetitions: int,
        activations: List[str],
        min_neurons: int,
        max_neurons: int,
        max_layers: int,
        stack_hidden_layers: bool,
        step_neurons: int,
        local_patience: int = 10,
        global_patience: int = 2,
        transform_data_strategy: str = "append_original_input",
        strategy_select_best: str = "architecture_median_best",
        loss_rel_tol: float = 0.05,
        min_improvement: float=0.001,
        device: str = "cuda",
        random_state: int = 0,
        logger: Any = None,
    ):
        if all_data_to_device and num_workers > 0:
            raise ValueError("num_workers must be 0 if all_data_to_device is True.")
        super().__init__()
        self.all_data_to_device = all_data_to_device
        self.loss_function = loss_function
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.repetitions = repetitions
        self.activations = activations
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.max_layers = max_layers
        self.stack_hidden_layers = stack_hidden_layers
        self.step_neurons = step_neurons
        self.local_patience = local_patience
        self.global_patience = global_patience
        self.transform_data_strategy = transform_data_strategy
        self.strategy_select_best = strategy_select_best
        self.loss_rel_tol = loss_rel_tol
        self.min_improvement = min_improvement
        self.device: str = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.hidden_neuron__model_id, self.output__model_id, self.output__architecture_id = build_model_ids(
            self.repetitions,
            self.activations,
            self.min_neurons,
            self.max_neurons,
            self.step_neurons,
        )


        self.best_model = None

        self.num_trained_mlps = 0
    

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
            pin_memory=not self.all_data_to_device,
        )

        return dataloader

    def _get_best_mlp(
        self, train_dataloader: DataLoader, validation_dataloader: DataLoader
    ):
        loss_function = deepcopy(self.loss_function)
        loss_function.reduction = "none"

        in_features = train_dataloader.dataset[0][0].shape[0]
        out_features = len(train_dataloader.dataset.tensors[1].unique())

        # profiler = pyinstrument.Profiler()
        # profiler.start()
        start = perf_counter()
        pmlps = ParallelMLPs(
            in_features=in_features,
            out_features=out_features,
            hidden_neuron__model_id=self.hidden_neuron__model_id,
            output__model_id=self.output__model_id,
            output__architecture_id=self.output__architecture_id,
            activations=self.activations,
            bias=True,
            device=self.device,
            logger=self.logger,
        ).to(self.device)
        end = perf_counter()
        # profiler.stop()
        # profiler.output_html()

        self.num_trained_mlps += pmlps.num_unique_models

        self.logger.info(
            f"Created ParallelMLPs in {end-start} seconds with {pmlps.num_unique_models}, starting with {self.min_neurons} neurons to {self.max_neurons} and step {self.step_neurons}, with activations {self.activations}, repeated {self.repetitions}"
        )

        optimizer: Optimizer = config.create_optimizer(self.optimizer_name, self.learning_rate, pmlps.parameters())

        train_loss = Accumulator(
            name="train_loss",
            objective=ObjectiveEnum.MINIMIZATION,
            reduction_fn=lambda individual_losses: torch.mean(
                torch.cat(individual_losses, dim=0), dim=0
            ),
        )
        epoch_validation_loss = Accumulator(
            name="validation_loss",
            objective=ObjectiveEnum.MINIMIZATION,
            reduction_fn=lambda individual_losses: torch.mean(
                torch.cat(individual_losses, dim=0), dim=0
            ),
            min_relative_improvement=self.min_improvement
        )
        best_validation_loss = torch.tensor(float("inf"))
        model__best_validation_loss = torch.ones(pmlps.num_unique_models) * float("inf")
        current_patience = torch.zeros(pmlps.num_unique_models)
        total_local_resets = 0

        t = tqdm(range(self.num_epochs))
        for epoch in t:
            self.train()
            train_loss.reset(reset_best=False)
            reduced_train_loss = self.execute_loop(
                train_dataloader, loss_function, optimizer, pmlps, train_loss
            )  # [num_models]
            epoch_best_train_loss = reduced_train_loss.min().cpu().item()

            self.eval()
            epoch_validation_loss.reset(reset_best=False)
            with torch.no_grad():
                reduced_validation_loss = self.execute_loop(
                    validation_dataloader, loss_function, None, pmlps, epoch_validation_loss
                )  # [num_models]

                detached_reduced_train_loss = reduced_train_loss.detach().cpu()
                detached_reduced_validation_loss = reduced_validation_loss.detach().cpu()

            # TODO: ajustar para model__best_validation_loss usar o improved global (ja que estou resetando o best daquele modelo especifico quando paciencia estoura)
            model__best_validation_loss[epoch_validation_loss.improved] = detached_reduced_validation_loss[epoch_validation_loss.improved]

            current_best_validation_loss, current_best_model_id = self._get_bests(pmlps, model__best_validation_loss)

            percentage_diff, better_model = helpers.has_improved(current_best_validation_loss, best_validation_loss, self.min_improvement, epoch_validation_loss.objective)

            # if current_best_validation_loss < best_validation_loss:
            if better_model:
                best_validation_loss = current_best_validation_loss
                best_mlp = pmlps.extract_mlp(current_best_model_id)

            current_patience[~epoch_validation_loss.improved] += 1
            current_patience[epoch_validation_loss.improved] = 0

            model_ids_to_reset = torch.where(current_patience > self.local_patience)[0]

            num_models_to_reset = len(model_ids_to_reset)
            if num_models_to_reset > 0:
                # TODO: acompanhar resultado final de cada arquitetura para fazer uma media (rolling avg?) da performance
                # daquela arquitetura (ja que varias simulacoes dela pode acontecer aqui)
                epoch_validation_loss.reset_best_from_ids(model_ids_to_reset)
                total_local_resets += num_models_to_reset
                pmlps.reset_parameters(model_ids_to_reset)
                # TODO: reset optimizer states
                current_patience[model_ids_to_reset] = 0

            t.set_postfix(
                train_loss=epoch_best_train_loss,
                best_validation_loss=best_validation_loss,
            )

            wandb.log(
                {"train/loss/avg": detached_reduced_train_loss.mean(), "epoch": epoch}
            )
            wandb.log(
                {"train/loss/min": detached_reduced_train_loss.min(), "epoch": epoch}
            )
            # wandb.log(
            #     {
            #         "train/losses": wandb.Histogram(detached_reduced_train_loss.cpu()),
            #         "epoch": epoch,
            #     }
            # )
            # data = [
            #     [x, y]
            #     for (x, y) in zip(
            #         detached_reduced_train_loss, range(pmlps.num_unique_models)
            #     )
            # ]
            # table = wandb.Table(data=data, columns=["train_loss", "hidden_neurons"])
            # wandb.log(
            #     {
            #         "my_custom_id": wandb.plot.scatter(
            #             table, "train_loss", "hidden_neurons"
            #         )
            #     }
            # )
            wandb.log(
                {
                    "validation/loss/avg": detached_reduced_validation_loss.mean(),
                    "epoch": epoch,
                }
            )
            wandb.log(
                {
                    "validation/loss/min": detached_reduced_validation_loss.min(),
                    "epoch": epoch,
                }
            )
            # wandb.log(
            #     {
            #         "validation/losses": wandb.Histogram(
            #             detached_reduced_validation_loss.cpu()
            #         ),
            #         "epoch": epoch,
            #     }
            # )

        self.num_trained_mlps += total_local_resets
        self.logger.info(f"Reset {total_local_resets} mlps to construct this layer. num_trained_mlps so far: {self.num_trained_mlps}")

        return best_mlp, best_validation_loss

    def _get_bests(self, pmlps, model__validation_loss):
        if self.strategy_select_best == StrategySelectBestEnum.GLOBAL_BEST:
            best_validation_loss = model__validation_loss.min().cpu().item()
            best_model_id = helpers.min_ix_argmin(
                        model__validation_loss, pmlps.model_id__num_hidden_neurons#, ignore_zeros=True
                    )
        elif self.strategy_select_best == StrategySelectBestEnum.ARCHITECTURE_MEDIAN_BEST:
            df = pd.DataFrame({'model_id': pmlps.output__model_id.cpu(), 'architecture_id': pmlps.output__architecture_id.cpu(), "validation_loss": model__validation_loss.cpu()})
            best_architecture_id = df.groupby(["architecture_id"]).median()["validation_loss"].argmin()
            df_architectures = df[df["architecture_id"] == best_architecture_id]
            best_validation_loss = df_architectures["validation_loss"].min()
            best_model_id = df_architectures[df_architectures['validation_loss'] == best_validation_loss]["model_id"].to_list()[0]
        else:
            raise RuntimeError(f"strategy_select_best {self.strategy_select_best} not recognized.")

        return best_validation_loss, best_model_id

    def execute_loop(self, dataloader, loss_function, optimizer, pmlps, accumulator):
        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            if optimizer:
                optimizer.zero_grad()

            outputs = pmlps(x)  # [batch_size, num_models, out_features]
            individual_losses = pmlps.calculate_loss(
                loss_func=loss_function, preds=outputs, target=y
            )
            accumulator.update(individual_losses.detach())

            loss = individual_losses.mean(
                0
            ).sum()  # [batch_size, num_models] -> [num_models] -> []

            if optimizer:
                loss.backward()
                optimizer.step()
        reduced_accumulator = accumulator.apply_reduction()

        return reduced_accumulator

    def _apply_forward_transform_data(
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

            current_train_x = self._transform_data(original_train_x, current_train_x)
            current_validation_x = self._transform_data(
                original_validation_x, current_validation_x
            )
        best_mlp.train()

        return current_train_x, current_validation_x

    def _transform_data(self, original_x, x):
        if self.transform_data_strategy == "append_original_input":
            x = torch.cat((original_x, x), dim=1)

        return x

    def __adjust_data(self, x, y=None):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.tensor(y)

            y = y.squeeze()

        return x, y

    def fit(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_validation: Tensor,
        y_validation: Tensor,
    ):

        eps = 1e-5
        x_train, y_train = self.__adjust_data(x_train, y_train)
        x_validation, y_validation = self.__adjust_data(x_validation, y_validation)
        nb_labels = len(y_train.unique())
        max_trivial_layers = 1

        if self.all_data_to_device:
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            x_validation = x_validation.to(self.device)
            y_validation = y_validation.to(self.device)

        current_train_x = x_train
        current_train_y = y_train

        current_validation_x = x_validation
        current_validation_y = y_validation

        global_best_validation_loss = float("inf")

        current_model = nn.Sequential()

        current_patience = 0
        current_layer_index = 0

        while (
            current_patience < self.global_patience and
            (
                self.max_layers is None or
                current_layer_index < self.max_layers
            )
        ):
            current_layer_index += 1
            train_dataloader = self._get_dataloader(current_train_x, current_train_y)
            validation_dataloader = self._get_dataloader(
                current_validation_x, current_validation_y
            )

            current_best_mlp, current_best_validation_loss = self._get_best_mlp(
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
            )

            current_best_mlp_nb_hidden = current_best_mlp[0].out_features 
            if current_best_mlp_nb_hidden <= nb_labels and len(current_model) > 0:
                if max_trivial_layers == 0:
                    self.logger.info(
                        f"Current best_mlp hidden neurons: {current_best_mlp_nb_hidden} <= nb_labels {nb_labels}. Stop appending layers."
                    )
                    break

                max_trivial_layers -= 1

            current_model.add_module(name=f"{len(current_model)}", module=current_best_mlp)

            # percentage_of_global_best_loss = current_best_validation_loss / (global_best_validation_loss+eps)
            # better_model = percentage_of_global_best_loss < (1-self.min_improvement)
            percentage_best_validation_loss, better_model = helpers.has_improved(current_best_validation_loss, global_best_validation_loss, self.min_improvement, ObjectiveEnum.MINIMIZATION, eps)

            self.logger.info(
                f"percentage_of_global_best_loss({percentage_best_validation_loss}) = current_best_validation_loss({current_best_validation_loss})/global_best_validation_loss({global_best_validation_loss}) < {1-self.min_improvement} (={better_model})."
            )

            if better_model:
                self.best_model = deepcopy(current_model)
                self.logger.info(
                    f"Improved validation loss from {global_best_validation_loss} to {current_best_validation_loss}. Setting current_patience=0. Current best model with {len(self.best_model)-1} layers: ({self.best_model})."
                )
                global_best_validation_loss = current_best_validation_loss
                current_patience = 0
            else:
                current_patience += 1
                self.logger.info(
                    f"No improvement. current_patience={current_patience}."
                )

            if current_best_validation_loss < eps:
                self.logger.info(f"current_best_validation_loss ({current_best_validation_loss}) < eps ({eps}). Stopping fit.")
                break

            if self.stack_hidden_layers:
                current_best_mlp = current_best_mlp[:-1]
                current_model[-1] = current_model[-1][:-1]

            current_train_x, current_validation_x = self._apply_forward_transform_data(
                x_train, x_validation, current_train_x, current_validation_x, current_best_mlp
            )

    def get_best_model_arch(self):
        arch = [self.best_model[0][0].in_features]
        for sequential in self.best_model:
            if hasattr(sequential[0], "out_features"):
                arch.append(sequential[0].out_features)
        arch.append(self.best_model[-1][2].out_features)
        return arch

    def predict(self, x: Tensor):
        x, _ = self.__adjust_data(x, None)
        x = x.to(self.device)
        h = x
        total_layers = len(self.best_model)
        for i, layer in enumerate(self.best_model):
            h = layer(h)

            if i < total_layers - 1:
                h = self._transform_data(x, h)

        return h
