from collections import Counter
from autoconstructive.utils.accumulators import ObjectiveEnum
from experiment_utils import assess_model

import pymcdm
from pathlib import Path
from kneed import KneeLocator
import heapq
import multiprocessing
from functools import partial
from copy import deepcopy
from autoconstructive.utils.multi_confusion_matrix import MultiConfusionMatrix
from conf.config import MAP_ACTIVATION
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from queue import PriorityQueue
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
from enum import Enum

# import pyinstrument

from conf import config
from time import perf_counter
import wandb
import numpy as np
import logging
import math
from autoconstructive.utils import helpers
from autoconstructive.utils.accumulators import Objective
from autoconstructive.autoconstructive_enums import ObjectiveEnum
from typing import Any, List, Optional, Type
from torch.functional import Tensor
from autoconstructive.model.parallel_mlp import MLP, ParallelMLPs, build_model_ids
from torch import nn
import torch
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

from conf.config import MAP_ACTIVATION, create_optimizer
from experiment_utils import assess_model

eps = 1e-5


class StrategySelectBestEnum(str, Enum):
    GLOBAL_BEST = "global_best"
    ARCHITECTURE_MEDIAN_BEST = "architecture_median_best"  # select architecture: groupby(architecture).median => Get best model among selected architecture
    ARCHITECTURE_MEDIAN_MEDIAN = "architecture_median_median"  # select architecture: groupby(architecture).median => Get the median model among selected architecture


class MyModel(nn.Module):
    def __init__(
        self, module, is_ensemble=False, output_confidence=False, min_confidence=0.5
    ):
        super().__init__()
        self.module = module
        self.class_avg_activations = None
        self.class_std_activations = None
        self.use_activation_statistics = False
        self.is_ensemble = is_ensemble
        self.output_confidence = output_confidence
        if not output_confidence:
            min_confidence = None
        self.min_confidence = min_confidence
        if hasattr(self.module, "device"):
            self.device = self.module.device
        else:
            self.device = self.module[0].device

    def __repr__(self):
        return self.module.__repr__()

    @property
    def out_features(self):
        if hasattr(self.module, "out_features"):
            return self.module.out_features
        else:
            return self.module[0].out_features

    def calculate_activation_statistics(self, train_dataloader):
        self.use_activation_statistics = True
        old_training = self.training
        self.eval()
        predictions = []
        ys = []
        with torch.no_grad():
            for (train_idx, x, y) in train_dataloader:
                x = x.to(self.device)

                predictions.append(self.module(x))
                ys.append(y)

            predictions = torch.cat(predictions)
            ys = torch.cat(ys)

        self.class_avg_activations = []
        self.class_std_activations = []

        for lb in sorted(ys.unique().cpu().numpy()):
            rows_lb = ys == lb
            self.class_avg_activations.append(predictions[rows_lb].mean())
            self.class_std_activations.append(predictions[rows_lb].std())

    def maybe_abstain_models(self, confidences):
        if self.min_confidence is not None:
            abstain = confidences.max(-1)[0] < self.min_confidence
            mask_samples_all_models_abstained = abstain.prod(0).bool()
            # If no one has confidence, better trust in all models.
            abstain[:, mask_samples_all_models_abstained] = False
            confidences[abstain, :] = float("nan")
        return confidences

    def __getitem__(self, i):
        return self.module[i]

    def forward(self, x):
        if self.is_ensemble:
            model__predictions = []
            for module in self.module:
                model__predictions.append(module(x))
            x = torch.stack(model__predictions, dim=0)

            x = self.maybe_abstain_models(x)

            x = x.nanmean(0)
        elif isinstance(self.module, MLP):
            x = self.module(x)

            if self.output_confidence:
                confidence = []
                for lb in range(x.shape[1]):
                    confidence.append(
                        torch.distributions.Normal(
                            self.class_avg_activations[lb],
                            self.class_std_activations[lb],
                        ).cdf(x[:, lb])
                    )
                # confidence = torch.stack(confidence, dim=1)
                # abstain = confidence.max(1)[0] < self.min_confidence
                # confidence[abstain, :] = float("nan")
                # x = confidence
        else:
            for module in self.module:
                x = module(x)

        return x


class AutoConstructiveModel(nn.Module):
    def __init__(
        self,
        all_data_to_device: bool,
        loss_function: nn.Module,
        optimizer_name: str,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        drop_samples: Optional[float],
        input_perturbation: Optional[str],
        regularization_gamma: Optional[float],
        monitored_metric: str,
        monitored_objective: ObjectiveEnum,
        pareto_frontier: bool,
        find_num_neurons_first: bool,
        mcdm_weights: Optional[List[float]],
        num_workers: int,
        repetitions: int,
        repetitions_for_best_neuron: int,
        activations: List[str],
        topk: Optional[int],
        topk_architecture: Optional[int],
        output_confidence: bool,
        min_confidence: float,
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
        min_improvement: float = 0.001,
        device: str = "cuda",
        random_state: int = 0,
        logger: Any = None,
        debug_test: bool = False,
        reset_exhausted_models: bool = False,
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
        self.drop_samples = drop_samples
        self.input_perturbation = input_perturbation
        self.regularization_gamma = regularization_gamma
        self.monitored_metric = monitored_metric
        self.monitored_objective = monitored_objective
        self.pareto_frontier = pareto_frontier
        self.find_num_neurons_first = find_num_neurons_first
        self.mcdm_weights = mcdm_weights
        self.num_workers = num_workers
        self.repetitions = repetitions
        self.repetitions_for_best_neuron = repetitions_for_best_neuron
        self.activations = activations
        self.cross_validation = True
        self.topk = topk
        self.topk_architecture = topk_architecture
        self.output_confidence = output_confidence
        self.min_confidence = min_confidence
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
        self.debug_test = debug_test
        self.reset_exhausted_models = reset_exhausted_models

        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
            logger.warn("No CUDA found. Falling back bo CPU.")

        self.device: str = device
        self.random_state = random_state
        torch.manual_seed(random_state)
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        (
            self.hidden_neuron__model_id,
            self.output__model_id,
            self.output__architecture_id,
            self.output__repetition,
            self.output__activation,
        ) = build_model_ids(
            self.repetitions,
            self.activations,
            self.min_neurons,
            self.max_neurons,
            self.step_neurons,
        )

        self.best_model_sequential = None

        self.num_trained_mlps = 0

        self.validation_df = pd.DataFrame()

        self.train_indices = torch.zeros(())

    @property
    def is_ensemble(self):
        return self.topk is not None and self.topk > 1

    def get_models_df(
        self, training_loss, validation_loss, model_ids, append_to_validation=False
    ):
        self.pmlps.eval()
        with torch.no_grad():
            current_rows = {
                "model_id": self.pmlps.output__model_id[model_ids].cpu().numpy(),
                "architecture_id": self.pmlps.output__architecture_id[model_ids]
                .cpu()
                .numpy(),
                "num_neurons": self.pmlps.model_id__num_hidden_neurons[model_ids]
                .cpu()
                .numpy(),
                "regularization_term": self.pmlps.get_regularization_term(gamma=1, l=1)
                .cpu()
                .numpy(),
                "activation": self.pmlps.get_activation_from_model_id(model_ids),
                "training_loss": training_loss[model_ids].cpu().numpy(),
                "validation_loss": validation_loss[model_ids].cpu().numpy(),
                "model_id": model_ids,
                "start_epoch": self.model__start_epoch[model_ids].cpu().numpy(),
                "end_epoch": self.epoch,
            }

            def _assess_model(dataloader, model_ids):
                # results = []
                results = {}
                if dataloader:
                    multi_cm = MultiConfusionMatrix(
                        len(model_ids), self.pmlps.out_features, "cpu"
                    )
                    x_val, y_val = dataloader.dataset[:]
                    y_preds = self.pmlps(
                        x_val.to(self.device)
                    ).cpu()  # [batch_size, num_unique_models, out_features]
                    multi_cm.update(y_preds, y_val)
                    # for model_id in model_ids:
                    #     r = assess_model(
                    #         y_preds[:, model_id, :].cpu().numpy(),
                    #         y_val.cpu().numpy(),
                    #         "",
                    #     )[0]
                    #     keys = list(r.keys())
                    #     keys.remove("__confusion_matrix")
                    #     for k in keys:
                    #         if k not in results:
                    #             results[k] = []
                    #         # results.append(r[0]["__overall_acc"])
                    #         results[k].append(r[k])
                    results = multi_cm.calculated_metrics
                return results

            # results = {}
            # for dataloader in [
            #     self.train_dataloader,
            #     self.validation_dataloader,
            #     self.test_dataloader,
            # ]:
            #     if dataloader is None:
            #         continue

            #     x_val, y_val = dataloader.dataset[:]
            #     y_preds = self.pmlps(
            #         x_val.to(self.device)
            #     ).cpu()  # [batch_size, num_unique_models, out_features]
            #     with multiprocessing.Pool(processes=10) as p:
            #         r = p.map(
            #             partial(
            #                 helpers.debug_assess_model, logits=y_preds, y_labels=y_val
            #             ),
            #             model_ids,
            #         )
            #         results[dataloader] = [e[0]["__overall_acc"] for e in r]

            results = {}
            results["train"] = _assess_model(self.train_dataloader, model_ids)
            results["valid"] = _assess_model(self.validation_dataloader, model_ids)
            results["test"] = _assess_model(self.test_dataloader, model_ids)

            for split in results.keys():
                split_results = results[split]
                for k in split_results.keys():
                    current_rows[f"{split}_{k}"] = np.array(split_results[k])
            # if train_results:
            #     current_rows["train_overall_acc"] = train_results

            # if val_results:
            #     current_rows["valid_overall_acc"] = val_results

            # if test_results:
            #     current_rows["test_overall_acc"] = test_results

            # if self.validation_dataloader:
            #     x_val, y_val = self.validation_dataloader.dataset[:]
            #     y_preds=self.pmlps(x_val) # [batch_size, num_unique_models, out_features]
            # for model_id in range(y_preds.shape[1]):
            #     assess_model(y_preds[:, model_id, :], y_val, "valid")

        # if test_results:
        #     current_rows
        current_df = pd.DataFrame(current_rows)
        if append_to_validation:
            self.validation_df = pd.concat((self.validation_df, current_df))

        return current_df

    def _get_dataloader(self, x: Tensor, y: Tensor, shuffle=True):
        indices = torch.arange(x.shape[0])
        if self.all_data_to_device:
            x = x.to(self.device)
            y = y.to(self.device)
            indices = indices.to(self.device)

        dataset = TensorDataset(indices, x, y)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=not self.all_data_to_device,
        )

        return dataloader

    # @profile
    def _get_best_mlps(
        self,
        hidden_neuron__model_id: List[int],
        output__model_id: List[int],
        output__architecture_id: List[int],
        output__repetition: List[int],
        activations: List[str],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader],
        from_find_num_neurons: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        loss_function = deepcopy(self.loss_function)
        loss_function.reduction = "none"

        in_features = train_dataloader.dataset[0][1].shape[0]
        out_features = len(train_dataloader.dataset.tensors[-1].unique())

        # profiler = pyinstrument.Profiler()
        # profiler.start()
        start = perf_counter()
        # output__repetition = (
        #     torch.arange(self.repetitions)
        #     .repeat_interleave(len(np.unique(output__model_id)) // self.repetitions)
        #     .tolist()
        # )
        self.pmlps = ParallelMLPs(
            in_features=in_features,
            out_features=out_features,
            hidden_neuron__model_id=hidden_neuron__model_id,
            output__model_id=output__model_id,
            output__architecture_id=output__architecture_id,
            output__repetition=output__repetition,
            drop_samples=self.drop_samples,
            input_perturbation_strategy=self.input_perturbation,
            activations=activations,
            bias=True,
            device=self.device,
            logger=self.logger,
        ).to(self.device)
        end = perf_counter()
        # profiler.stop()
        # profiler.output_html()

        self.num_trained_mlps += self.pmlps.num_unique_models

        self.logger.info(
            f"Created ParallelMLPs in {end-start} seconds with {self.pmlps.num_unique_models}, starting with {self.min_neurons} neurons to {self.max_neurons} and step {self.step_neurons}, with activations {self.activations}, repeated {self.repetitions}"
        )

        optimizer: Optimizer = config.create_optimizer(
            self.optimizer_name, self.learning_rate, self.pmlps.parameters()
        )

        accumulators = {
            "train_loss": Objective(
                "train_loss", ObjectiveEnum.MINIMIZATION, reduction_fn=None
            ),
            "validation_loss": Objective(
                "validation_loss", ObjectiveEnum.MINIMIZATION, reduction_fn=None
            ),
            "holdout_loss": Objective(
                "holdout_loss", ObjectiveEnum.MINIMIZATION, reduction_fn=None
            ),
            "test_loss": Objective(
                "test_loss", ObjectiveEnum.MINIMIZATION, reduction_fn=None
            ),
            "monitored_metric": Objective(
                self.monitored_metric,
                self.monitored_objective,
                reduction_fn=None,
            ),
        }

        model__global_best_metric = Objective(
            name=self.monitored_metric,
            objective=self.monitored_objective,
            reduction_fn=None,
        )

        self.current_patience = torch.zeros(self.pmlps.num_unique_models)
        self.model__start_epoch = torch.zeros(self.pmlps.num_unique_models)
        self.total_local_resets = 0

        best_mlps = []
        mlps = {}
        metrics_max = None

        t = tqdm(range(self.num_epochs))
        pmlps_df = pd.DataFrame()

        train_multi_metrics = MultiConfusionMatrix(
            self.pmlps.num_unique_models,
            self.pmlps.out_features,
            # "cpu",  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.
            self.device,
            self.pmlps.output__model_id,
        )
        validation_multi_metrics = MultiConfusionMatrix(
            self.pmlps.num_unique_models,
            self.pmlps.out_features,
            # "cpu",  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.
            self.device,
            self.pmlps.output__model_id,
        )
        holdout_multi_metrics = MultiConfusionMatrix(
            self.pmlps.num_unique_models,
            self.pmlps.out_features,
            # "cpu",  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.
            self.device,
            self.pmlps.output__model_id,
        )
        test_multi_metrics = MultiConfusionMatrix(
            self.pmlps.num_unique_models,
            self.pmlps.out_features,
            # "cpu",  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.
            self.device,
            self.pmlps.output__model_id,
        )
        for epoch in t:
            self.epoch = epoch

            not_exhausted_models = (self.current_patience < self.local_patience).to(
                self.device
            )
            num_not_exhausted_models = not_exhausted_models.sum()
            if num_not_exhausted_models == 0:
                self.logger.info(
                    f"All models are exhausted. Stopping training at epoch {epoch}."
                )
                break

            self.train()

            # accumulators["train_loss"].reset(reset_best=False)
            train_multi_metrics.reset_cm()
            models_train_loss, train_multi_metrics = self.execute_loop(
                self.train_dataloader,
                loss_function,
                optimizer,
                not_exhausted_models=not_exhausted_models,
                phase="train",
                # return_multi_metrics=True,
                multi_cm=train_multi_metrics,
            )  # [num_models]
            # models_train_loss.apply_reduction()
            # accumulators["train_loss"].set_values(models_train_loss.current_reduction)

            self.eval()

            accumulators["validation_loss"].reset(reset_best=False)
            accumulators["monitored_metric"].reset(reset_best=False)

            # Monitoring multi_metrics if overall_acc or is the last epoch
            return_multi_metrics = (
                self.monitored_metric != "validation_loss"
                or epoch == self.num_epochs - 1
            )
            # return_multi_metrics = True

            with torch.inference_mode():
                validation_multi_metrics.reset_cm()
                models_validation_loss, validation_multi_metrics = self.execute_loop(
                    dataloader=self.train_dataloader,  # passing train but we mask it inside
                    loss_function=loss_function,
                    optimizer=None,
                    not_exhausted_models=None,
                    multi_cm=validation_multi_metrics,
                    phase="validation",
                )  # [num_models]
                models_validation_loss.apply_reduction()
                accumulators["validation_loss"].set_values(
                    models_validation_loss.current_reduction
                )

                if validation_multi_metrics is not None:
                    accumulators["monitored_metric"].set_values(
                        validation_multi_metrics.calculated_metrics[
                            self.monitored_metric
                        ]
                    )

                model__global_best_metric.reset()
                model__global_best_metric.set_values(
                    accumulators["monitored_metric"].current_reduction
                )
                best_improved_model_ids = model__global_best_metric.get_best_k_ids(
                    -1, only_improved=True
                )
                best_improved_model_ids_numpy = best_improved_model_ids.cpu().numpy()

                best_metrics = model__global_best_metric.best[best_improved_model_ids]

                # if model__global_best_metric.objective == ObjectiveEnum.MAXIMIZATION:
                #     best_metrics = best_metrics * (-1) # because pareto expect "costs" (minimization problem)

                holdout_multi_metrics.reset_cm()
                accumulators["holdout_loss"].reset(reset_best=True)
                models__holdout_loss, holdout_multi_metrics = self.execute_loop(
                    dataloader=self.validation_dataloader,
                    loss_function=loss_function,
                    optimizer=None,
                    not_exhausted_models=None,
                    multi_cm=holdout_multi_metrics,
                    phase="holdout",
                )  # [num_models]
                models__holdout_loss.apply_reduction()
                accumulators["holdout_loss"].set_values(
                    models__holdout_loss.current_reduction
                )

                if test_dataloader:
                    test_multi_metrics.reset_cm()
                    accumulators["test_loss"].reset()
                    models__test_loss, test_multi_metrics = self.execute_loop(
                        dataloader=test_dataloader,
                        loss_function=loss_function,
                        optimizer=None,
                        not_exhausted_models=None,
                        multi_cm=test_multi_metrics,
                        phase="test",
                    )  # [num_models]
                    models__test_loss.apply_reduction()
                    accumulators["test_loss"].set_values(
                        models__test_loss.current_reduction
                    )

                if len(best_improved_model_ids) > 0:
                    train_multi_metrics_df = train_multi_metrics.to_dataframe(
                        prefix="train_"
                    ).loc[best_improved_model_ids_numpy]
                    validation_multi_metrics_df = validation_multi_metrics.to_dataframe(
                        prefix="validation_"
                    ).loc[best_improved_model_ids_numpy]
                    holdout_multi_metrics_df = holdout_multi_metrics.to_dataframe(
                        prefix="holdout_"
                    ).loc[best_improved_model_ids_numpy]

                    epoch_bests_df = pd.DataFrame(
                        {
                            "selected": False,
                            "fold": self.output__kfold[best_improved_model_ids]
                            .cpu()
                            .numpy(),
                            "repetition": self.output__repetition[
                                best_improved_model_ids
                            ]
                            .cpu()
                            .numpy(),
                            "num_neurons": self.pmlps.model_id__num_hidden_neurons[
                                best_improved_model_ids
                            ]
                            .cpu()
                            .numpy(),
                            "weight_regularization": self.pmlps.get_regularization_term()[
                                best_improved_model_ids
                            ]
                            .cpu()
                            .numpy(),
                            "monitored_metric": best_metrics.cpu().numpy(),
                            "loss": models_validation_loss.current_reduction[
                                best_improved_model_ids_numpy
                            ]
                            .cpu()
                            .numpy(),
                            "repetition": self.pmlps.output__repetition[
                                best_improved_model_ids_numpy
                            ]
                            .cpu()
                            .numpy(),
                            "epoch": epoch,
                            "model_id": best_improved_model_ids_numpy,
                            "activation_name": self.pmlps.get_activation_from_model_id(
                                best_improved_model_ids_numpy
                            ),
                            "architecture_id": self.pmlps.get_architecture_ids_from_model_ids(
                                best_improved_model_ids
                            ),
                        },
                    ).set_index("model_id")
                    epoch_bests_df = pd.merge(
                        epoch_bests_df, train_multi_metrics_df, on=["model_id"]
                    )
                    epoch_bests_df = pd.merge(
                        epoch_bests_df, validation_multi_metrics_df, on=["model_id"]
                    )
                    epoch_bests_df = pd.merge(
                        epoch_bests_df, holdout_multi_metrics_df, on=["model_id"]
                    )

                    if test_dataloader:
                        test_multi_metrics_df = test_multi_metrics.to_dataframe(
                            prefix="test_"
                        )
                        test_multi_metrics_df = test_multi_metrics_df.loc[
                            best_improved_model_ids_numpy
                        ]
                        epoch_bests_df = pd.merge(
                            epoch_bests_df, test_multi_metrics_df, on=["model_id"]
                        )
                    epoch_bests_df["activation_name"] = epoch_bests_df[
                        "activation_name"
                    ].astype("str")

                    # Updating pmlps_df
                    if epoch == 0:
                        pmlps_df = epoch_bests_df
                    else:
                        pmlps_df.loc[epoch_bests_df.index, :] = epoch_bests_df
                        if from_find_num_neurons:
                            chosen_df = pmlps_df

                    to_add_or_update_pareto_mlps = dict(
                        zip(
                            best_improved_model_ids_numpy,
                            self.pmlps.extract_params_mlps(
                                best_improved_model_ids_numpy
                            ),
                            # self.pmlps.extract_mlps(best_improved_model_ids_numpy),
                        )
                    )

                    for model_id in to_add_or_update_pareto_mlps:
                        mlps[model_id] = to_add_or_update_pareto_mlps[model_id]

            self.current_patience[~accumulators["monitored_metric"].improved] += 1
            self.current_patience[accumulators["monitored_metric"].improved] = 0

            if self.reset_exhausted_models:
                with torch.inference_mode():
                    self._reset_exhausted_models(accumulators["monitored_metric"])

            t.set_postfix(
                perc_not_exhausted_models=num_not_exhausted_models
                / self.pmlps.num_unique_models,
                # train_loss=models_train_loss.current_reduction.min(),
                best_validation_loss=accumulators["validation_loss"].best.min(),
                monitored_metric=accumulators["monitored_metric"].best[
                    accumulators["monitored_metric"].get_best_k_ids(1)
                ],
            )

            # wandb.log(
            #     {
            #         "train/loss/avg": models_train_loss.current_reduction.detach()
            #         .mean()
            #         .cpu(),
            #         "epoch": epoch,
            #     }
            # )
            # wandb.log(
            #     {
            #         "train/loss/min": models_train_loss.current_reduction.detach()
            #         .min()
            #         .cpu(),
            #         "epoch": epoch,
            #     }
            # )
            wandb.log(
                {
                    "validation/loss/avg": models_validation_loss.current_reduction.mean().cpu(),
                    "epoch": epoch,
                }
            )
            wandb.log(
                {
                    "validation/loss/min": models_validation_loss.current_reduction.min().cpu(),
                    "epoch": epoch,
                }
            )
        self.num_trained_mlps += self.total_local_resets
        self.logger.info(
            f"Reset {self.total_local_resets} mlps to construct this layer. num_trained_mlps so far: {self.num_trained_mlps}"
        )

        chosen_df, chosen_model_ids = self.choose_model_ids(pmlps_df)
        # best_mlps = [e[1] for e in pareto_mlps if e[0] in chosen_model_ids]
        best_mlps = []
        for model_id in chosen_model_ids:
            mlp_args = mlps[model_id]

            hidden_weight = mlp_args["hidden_weight"]
            hidden_bias = mlp_args["hidden_bias"]
            out_weight = mlp_args["out_weight"]
            out_bias = mlp_args["out_bias"]

            hidden_layer = nn.Linear(
                in_features=hidden_weight.shape[1],
                out_features=hidden_weight.shape[0],
                device=self.device,
            )
            out_layer = nn.Linear(
                in_features=out_weight.shape[1],
                out_features=out_weight.shape[0],
                device=self.device,
            )

            with torch.no_grad():
                hidden_layer.weight[:, :] = hidden_weight
                hidden_layer.bias[:] = hidden_bias
                out_layer.weight[:, :] = out_weight
                out_layer.bias[:] = out_bias

            best_mlps.append(
                MLP(
                    hidden_layer=hidden_layer,
                    out_layer=out_layer,
                    activation=mlp_args["activation"],
                    model_id=mlp_args["metadata"]["model_id"],
                    metadata=mlp_args["metadata"],
                    device=self.device,
                ).to(self.device)
            )

        # best_mlps = [mlps[model_id] for model_id in chosen_model_ids]

        metrics_max = torch.tensor(chosen_df["monitored_metric"].max())

        print("Chosen df:")
        print(chosen_df)
        # Testing
        # if self.test_dataloader:
        #     validation_results = self.assess_mlps(
        #         pareto_mlps, ranked_pmlps_df, "validation"
        #     )
        #     test_results = self.assess_mlps(pareto_mlps, validation_results, "test")
        #     test_results.to_csv(
        #         f"results_df_debug_{self.current_layer_index}.csv",
        #     )

        return (
            best_mlps,
            metrics_max,
            chosen_df,
        )

    def choose_model_ids(self, pmlps_df):
        agg = "mean"

        # architecture_counters = {i: 0 for i in pmlps_df["architecture_id"]}
        # for arch_id in pmlps_df["architecture_id"]:
        #     architecture_counters[arch_id] += 1
        #     if architecture_counters[arch_id] == self.repetitions:
        #         best_arch_id = arch_id
        #         break
        # pmlps_df["best_arch_id"] = pmlps_df["architecture_id"] == best_arch_id

        grouped_pmlps = (
            pmlps_df.groupby(["architecture_id"])[
                list(set(pmlps_df.columns).difference({"activation_name"}))
            ]
            .agg([agg, "std"])
            .sort_values(
                by=[
                    # ("monitored_metric", agg),
                    ("holdout_overall_acc", agg),
                    # ("monitored_metric", "std"),
                    ("num_neurons", agg),
                ],
                ascending=[False, True],
            )
        ).reset_index()

        grouped_pmlps.to_csv(
            f"grouped_pmlps_{self.current_layer_index}.csv",
            float_format="{:f}".format,
        )

        pareto_variables = pmlps_df[
            ["num_neurons", "holdout_overall_acc", "epoch"]
        ].to_numpy()
        pareto_variables *= np.array(
            [1, -1, -1]
        )  # is_pareto_efficient works with minimization problems.
        pmlps_df["dominant_solution"] = helpers.is_pareto_efficient(pareto_variables)
        pmlps_df["mean_diffs"] = (
            (
                (
                    abs(pmlps_df["holdout_overall_acc"] - pmlps_df["train_overall_acc"])
                    + abs(
                        pmlps_df["holdout_overall_acc"]
                        - pmlps_df["validation_overall_acc"]
                    )
                    + abs(
                        pmlps_df["train_overall_acc"]
                        - pmlps_df["validation_overall_acc"]
                    )
                )
                / 3
            )
            # + (
            #     abs(
            #         pmlps_df["holdout_matthews_corrcoef"]
            #         - pmlps_df["train_matthews_corrcoef"]
            #     )
            #     + abs(
            #         pmlps_df["holdout_matthews_corrcoef"]
            #         - pmlps_df["validation_matthews_corrcoef"]
            #     )
            #     + abs(
            #         pmlps_df["train_matthews_corrcoef"]
            #         - pmlps_df["validation_matthews_corrcoef"]
            #     )
            # )
            # / 3
        ) / 1

        pmlps_df = pmlps_df.sort_values(
            by=["mean_diffs", "num_neurons", "holdout_overall_acc"],
            ascending=[True, True, False],
        )
        pmlps_df.to_csv(
            f"pmlps_{self.current_layer_index}.csv",
            float_format="{:f}".format,
        )

        pmlps_df = pmlps_df[
            (pmlps_df["epoch"] >= 5) & (pmlps_df["holdout_matthews_corrcoef"] > 0.05)
        ]
        pmlps_df.to_csv(
            f"pmlps_after_filtering_{self.current_layer_index}.csv",
            float_format="{:f}".format,
        )
        # pmlps_df = pmlps_df[pmlps_df["dominant_solution"]]

        # name, value, best, worst
        mcdm_tuples = [
            ("num_neurons", -1),
            ("epoch", 1),
            ("holdout_overall_acc", 1),
            # (("num_neurons", agg), -1),
            # (("num_epochs", agg), 1),
            # (("holdout_overall_acc", agg), 1),
            # (("holdout_overall_acc", "std"), -1),
            # (("loss", "median"), -1, 0, 1),
            # (("loss", "std"), -1, 0, 1),
            # (("loss", "std"), -1, None, None),
        ]
        ranked_pmlps_df_only_pareto = self.get_ranked_pmlps_df(
            pmlps_df, mcdm_tuples, only_pareto_solutions=True
        )
        ranked_pmlps_df_only_pareto.to_csv(
            f"ranked_pmlps_df_only_pareto_{self.current_layer_index}.csv"
        )
        ranked_pmlps_df = self.get_ranked_pmlps_df(
            pmlps_df, mcdm_tuples, only_pareto_solutions=False
        )
        ranked_pmlps_df.to_csv(f"ranked_pmlps_df_{self.current_layer_index}.csv")

        if self.pareto_frontier:
            ranked_pmlps_df = ranked_pmlps_df_only_pareto

        # Get top 1% num_neurons
        # ranked_pmlps_df = ranked_pmlps_df.iloc[: int(ranked_pmlps_df.shape[0] * 0.01)]
        ranked_pmlps_df = ranked_pmlps_df.sort_values(
            # by=["num_neurons", "holdout_overall_acc", "epoch"],
            # ascending=[True, False, False],
            # by=["test_overall_acc", "num_neurons", "epoch"],
            # ascending=[False, True, False],
            # by=["test_overall_acc", "num_neurons"], # politica 1
            # by=["holdout_overall_acc", "num_neurons"], # politica 2
            by=["validation_overall_acc", "num_neurons"],  # politica 2
            ascending=[False, True],
        )
        ranked_pmlps_df.to_csv(
            f"top_ranked_pmlps_df_{self.current_layer_index}.csv",
            float_format="{:f}".format,
        )
        print(f"ranked_pmlps_df top 0.01: {ranked_pmlps_df.head(10)}")

        topk = min(self.topk, ranked_pmlps_df.shape[0])
        chosen_df = ranked_pmlps_df.iloc[:topk, :].reset_index()
        chosen_model_ids = chosen_df["model_id"].tolist()
        return chosen_df, chosen_model_ids

    def assess_mlps(self, pareto_mlps, ranked_pmlps_df, dataloader_name):
        with torch.inference_mode():
            if dataloader_name == "validation":
                dataloader = self.validation_dataloader
            elif dataloader_name == "test":
                dataloader = self.test_dataloader
            elif dataloader_name == "train":
                dataloader = self.train_dataloader
            else:
                raise ValueError(
                    f"Unrecognized value for dataloader_name={dataloader_name}."
                )
            x_test, y_test = dataloader.dataset[:]
            x_test = x_test.to(self.device)
            y_test = y_test.cpu().numpy()

            results_dict_list = []
            for model_id in ranked_pmlps_df["model_id"]:
                mlp = [e[1] for e in pareto_mlps if e[0] == model_id][0]
                old_mode = mlp.training
                mlp.eval()
                results_dict = assess_model(
                    mlp(x_test).cpu().numpy(), y_test, dataloader_name
                )[0]
                results_dict["model_id"] = model_id
                results_dict_list.append(results_dict)
                mlp.train(old_mode)
            results = pd.DataFrame(results_dict_list)
            results = pd.merge(ranked_pmlps_df, results, on="model_id")

            return results

    def get_ranked_pmlps_df(
        self,
        pmlps_df,
        mcdm_tuples,
        theoretical_best=None,
        theoretical_worst=None,
        only_pareto_solutions=True,
        sort_by_rank=False,
    ):

        mcdm_keys = [k[0] for k in mcdm_tuples]

        types = [k[1] for k in mcdm_tuples]
        mcdm_method = pymcdm.methods.TOPSIS(pymcdm.normalizations.minmax_normalization)

        decision_matrix = pmlps_df[mcdm_keys].to_numpy()

        pareto_mask = helpers.is_pareto_efficient(decision_matrix)
        pmlps_df["dominant_solution"] = pareto_mask
        if only_pareto_solutions:
            pmlps_df = pmlps_df.loc[pareto_mask]
            decision_matrix = pmlps_df[mcdm_keys].to_numpy()

        if theoretical_best is not None:
            if theoretical_worst is None:
                raise ValueError(
                    "Both theoretical_best and theoretical_worst must be None or have values."
                )
            decision_matrix = np.vstack(
                (decision_matrix, theoretical_best, theoretical_worst)
            )

        if self.mcdm_weights is not None:
            weights = np.array(self.mcdm_weights)
        else:
            weights = pymcdm.weights.equal_weights(decision_matrix)
        # weights = np.array([0.5, 0.4, 0.1])
        ranks = mcdm_method(decision_matrix, weights, types)
        # removing best_and_worst_theoretical_mlps
        if theoretical_best is not None:
            ranks = ranks[:-2]

        pmlps_df["rank"] = ranks

        if sort_by_rank:
            pmlps_df = pmlps_df.sort_values(by=["rank"], ascending=False).reset_index()

        return pmlps_df

    def _reset_exhausted_models(self, monitored_metric):
        model_ids_to_reset = torch.where(self.current_patience > self.local_patience)[0]

        num_models_to_reset = len(model_ids_to_reset)
        if num_models_to_reset > 0:
            # TODO: acompanhar resultado final de cada arquitetura para fazer uma media (rolling avg?) da performance
            # daquela arquitetura (ja que varias simulacoes dela pode acontecer aqui)
            # if self.debug_test:
            #     self.append_to_validation_df(
            #         epoch_train_loss, epoch_validation_loss, model_ids_to_reset
            #     )
            monitored_metric.reset_best_from_ids(model_ids_to_reset)
            self.total_local_resets += num_models_to_reset
            self.pmlps.reset_parameters(model_ids_to_reset)
            self.model__start_epoch[model_ids_to_reset] = self.epoch
            # TODO: reset optimizer states
            self.current_patience[model_ids_to_reset] = 0

    def _get_bests(self, model__validation_loss):
        if self.strategy_select_best == StrategySelectBestEnum.GLOBAL_BEST:
            best_validation_loss = model__validation_loss.min().cpu().item()
            best_model_id = helpers.min_ix_argmin(
                model__validation_loss,
                self.pmlps.model_id__num_hidden_neurons,  # , ignore_zeros=True
            )
        elif (
            self.strategy_select_best == StrategySelectBestEnum.ARCHITECTURE_MEDIAN_BEST
        ):
            df = pd.DataFrame(
                {
                    "model_id": self.pmlps.output__model_id.cpu(),
                    "architecture_id": self.pmlps.output__architecture_id.cpu(),
                    "validation_loss": model__validation_loss.cpu(),
                }
            )
            best_architecture_id = (
                df.groupby(["architecture_id"]).median()["validation_loss"].argmin()
            )
            df_architectures = df[df["architecture_id"] == best_architecture_id]
            best_validation_loss = df_architectures["validation_loss"].min()
            best_model_id = df_architectures[
                df_architectures["validation_loss"] == best_validation_loss
            ]["model_id"].to_list()[0]
        else:
            raise RuntimeError(
                f"strategy_select_best {self.strategy_select_best} not recognized."
            )

        return best_validation_loss, best_model_id

    # @profile
    def execute_loop(
        self,
        dataloader,
        loss_function,
        optimizer,
        not_exhausted_models=None,
        # return_multi_metrics=False,
        multi_cm=None,
        phase="train",
    ):
        loss_accumulator = Objective(
            "loss", ObjectiveEnum.MINIMIZATION, reduction_fn="mean"
        )
        # multi_cm = None
        # if return_multi_metrics:
        #     multi_cm = MultiConfusionMatrix(
        #         self.pmlps.num_unique_models,
        #         self.pmlps.out_features,
        #         # "cpu",  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.
        #         "cuda",
        #         self.pmlps.output__model_id,
        #     )

        drop_samples = torch.zeros((self.batch_size, self.pmlps.num_unique_models)).to(
            self.device
        )

        current_mask = None
        for (indices, x, y) in dataloader:
            indices = indices.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)

            if optimizer:
                optimizer.zero_grad()

            outputs = self.pmlps(x)  # [batch_size, num_models, out_features]

            individual_losses = self.pmlps.calculate_loss(
                loss_func=loss_function, preds=outputs, target=y
            )

            if self.drop_samples is not None and optimizer is not None:
                if x.shape[0] != self.batch_size:
                    drop_samples = torch.zeros(
                        (x.shape[0], self.pmlps.num_unique_models)
                    ).to(self.device)
                individual_losses *= drop_samples.uniform_() > self.drop_samples

            if self.regularization_gamma is not None:
                regularization_term = self.pmlps.get_regularization_term(
                    gamma=self.regularization_gamma
                )
                individual_losses = individual_losses + regularization_term

            if self.cross_validation:
                if phase == "train":
                    current_mask = self.train_mask[indices, :]
                elif phase == "validation":
                    current_mask = ~self.train_mask[indices, :]

            if current_mask is not None:
                individual_losses = individual_losses * current_mask

            loss_accumulator.accumulate_values(
                individual_losses.detach(), mask=current_mask
            )

            if optimizer:
                # loss = individual_losses.mean(
                #     0
                # ).sum()  # [batch_size, num_models] -> [num_models] -> []

                # ignoring validation indices

                if self.cross_validation:
                    loss = individual_losses.sum(0) / current_mask.sum(0)
                else:
                    loss = individual_losses.mean(
                        0
                    )  # [batch_size, num_models] -> [num_models]
                # if not_exhausted_models is not None:
                #     loss = loss * not_exhausted_models

                loss = loss.sum()  #  [num_models]-> []

                loss.backward()
                optimizer.step()

                loss_accumulator.accumulate_values(
                    individual_losses.detach(), mask=current_mask
                )

                self.pmlps.enforce_input_perturbation()

            # if return_multi_metrics:
            # if current_mask is not None:
            #     current_mask = current_mask.cpu()

            multi_cm.update(
                outputs, y, current_mask
            )  # TODO: change to CUDA when https://github.com/pytorch/pytorch/issues/72053 is fixed.

        return loss_accumulator, multi_cm

    def _apply_forward_transform_data(
        self,
        original_train_x,
        original_validation_x,
        original_test_x,
        current_train_x,
        current_validation_x,
        current_test_x,
        best_mlp,
    ):
        best_mlp.eval()
        with torch.no_grad():
            current = {
                "train": current_train_x,
                "validation": current_validation_x,
                "test": current_test_x,
            }
            original = {
                "train": original_train_x,
                "validation": original_validation_x,
                "test": original_test_x,
            }
            for k in current:
                if current[k] is None:
                    continue

                dataloader = self._get_dataloader(
                    current[k], torch.arange(current[k].shape[0]), shuffle=False
                )
                new_data = []
                for indices, x, _ in dataloader:
                    x = x.to(self.device)
                    new_data.append(best_mlp(x))
                current[k] = self._transform_data(
                    original[k], torch.cat(new_data).to(original[k].device)
                )

        best_mlp.train()

        return current["train"], current["validation"], current["test"]

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

    def find_num_neurons(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_validation: Tensor,
        y_validation: Tensor,
        x_test: Optional[Tensor],
        y_test: Optional[Tensor],
    ):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)

        results_df = None
        x = torch.cat((x_train, x_validation))
        y = torch.cat((y_train, y_validation))

        for i, (train_indices, validation_indices) in enumerate(
            skf.split(x.cpu(), y.cpu())
        ):
            x_train = x[train_indices]
            y_train = y[train_indices]
            x_validation = x[validation_indices]
            y_validation = y[validation_indices]

            if self.all_data_to_device:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                x_validation = x_validation.to(self.device)
                y_validation = y_validation.to(self.device)

            train_dataloader = self._get_dataloader(x_train, y_train)
            validation_dataloader = self._get_dataloader(x_validation, y_validation)

            _, _, current_df = self._get_best_mlps(
                hidden_neuron__model_id=self.hidden_neuron__model_id,
                output__model_id=self.output__model_id,
                output__architecture_id=self.output__architecture_id,
                output__repetition=self.output__repetition,
                activations=self.activations,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                test_dataloader=None,
                from_find_num_neurons=True,
            )

            current_df["k-fold"] = i

            if results_df is None:
                results_df = current_df
            else:
                results_df = pd.concat((results_df, current_df))

        results_df.to_csv(
            f"find_num_neurons_results_df_current_layer_index_{self.current_layer_index}.csv"
        )
        # activation_name is here only to appear at the final dataframe, once architecture_id has a 1-1 match with activation_name
        metric = "<lambda_0>"
        pmlps_df = (
            results_df.groupby(["architecture_id", "activation_name"])
            .agg(["mean", "median", "std", "count", lambda x: x.quantile(0.95)])
            .sort_values(by=("loss", metric))
        ).reset_index()
        helpers.save_flattend_df(
            pmlps_df,
            f"find_num_neurons_pmlps_df_current_layer_index_{self.current_layer_index}.csv",
        )
        method = "pareto"

        if method == "pareto":
            # name, value, best, worst
            min_neurons = self.pmlps.model_id__num_hidden_neurons.min().item()
            max_neurons = self.pmlps.model_id__num_hidden_neurons.max().item()

            # name, value, best, worst
            mcdm_tuples = [
                (("num_neurons", "median"), -1, min_neurons, max_neurons),
                (("metrics", "median"), -1, -1, 0),
                # (("loss", "std"), -1, None, None),
            ]
            mcdm_keys = [k[0] for k in mcdm_tuples]
            theoretical_best = np.array([[m[2] for m in mcdm_tuples]])
            theoretical_worst = np.array([[m[3] for m in mcdm_tuples]])

            pmlps_df = pmlps_df.drop(columns="dominant_solution")
            pmlps_df = self.get_ranked_pmlps_df(
                pmlps_df,
                mcdm_tuples=mcdm_tuples,
                # theoretical_best=theoretical_best,  # theoretical_best,
                # theoretical_worst=theoretical_worst,  # theoretical_worst,
                theoretical_best=None,
                theoretical_worst=None,
                only_pareto_solutions=True,
            )

            helpers.save_flattend_df(
                pmlps_df,
                f"find_num_neurons_pmlps_df_onlyl_pareto_current_layer_index_{self.current_layer_index}.csv",
            )

            # # Testing
            # if self.test_dataloader:
            #     validation_results = self.assess_mlps(
            #         pareto_mlps, ranked_pmlps_df, "validation"
            #     )
            #     test_results = self.assess_mlps(pareto_mlps, validation_results, "test")
            #     test_results.to_csv(
            #         f"results_df_debug_{self.current_layer_index}.csv",
            #     )

            print(pmlps_df)

            best = pmlps_df.iloc[0:1, :].reset_index()

        else:
            num_models = pmlps_df["model_id"]["count"].values[0]
            pmlps_df = pd.DataFrame()
            for activation in pmlps_df["activation_name"].unique():
                tmp_df = results_df[results_df["activation_name"] == activation]

                k = KneeLocator(
                    x=tmp_df["num_neurons"],
                    y=tmp_df["loss"],
                    curve="convex",
                    direction="decreasing",
                    interp_method="polynomial",
                    polynomial_degree=7,
                )

                pmlps_df = pd.concat(
                    (pmlps_df, tmp_df[tmp_df["num_neurons"] == k.knee])
                )
            best = (
                pmlps_df.groupby(["architecture_id", "activation_name"])
                .median()
                .sort_values(by="loss")
                .iloc[0:1, :]
                .reset_index()
            )

        # counter = {i: 0 for i in self.output__architecture_id}
        # for index, row in results_df.sort_values(by="loss").iterrows():
        #     architecture_id = row["architecture_id"]
        #     counter[architecture_id] += 1
        #     if counter[architecture_id] == num_models:
        #         best_architecture_id = architecture_id
        #         break

        # best = grouped_df[grouped_df["architecture_id"] == best_architecture_id]

        # best = grouped_df[
        #     grouped_df[("loss", metric)] == grouped_df[("loss", metric)].min()
        # ].reset_index()
        # num_neurons = best["num_neurons"]["mean"].item()
        self.logger.info(pmlps_df)
        # grouped_df = results_df.groupby(["architecture_id", "activation_name"]).mean()
        # # best_architecture_id = grouped_df[
        # #     grouped_df["loss"] == grouped_df["loss"].min()
        # # ].index.item()
        # best = grouped_df[grouped_df["loss"] == grouped_df["loss"].min()].reset_index()

        # model_id = (results_df[results_df['architecture_id'] == best_architecture_id]).index.min()

        # best_num_hidden_neurons = self.pmlps.model_id__num_hidden_neurons[model_id]
        # activation =
        # num_neurons = int(best["num_neurons"].item())
        num_neurons = int(best[("num_neurons", "median")].item())
        activation_name = [MAP_ACTIVATION[best["activation_name"].item()]()]
        self.logger.info(
            f"best num neurons: {num_neurons}, best activation: {activation_name}"
        )

        return num_neurons, activation_name

    def _generate_model_train_mask(self, num_unique_models, y, train_fold_ids):
        train_mask = torch.zeros((y.shape[0], num_unique_models)).bool()
        model__fold = torch.zeros(num_unique_models).int()
        if train_fold_ids is None:
            splitter = StratifiedKFold(
                n_splits=self.repetitions, shuffle=True, random_state=self.random_state
            )

            step = num_unique_models // self.repetitions
            iterator = splitter.split(y, y)
        else:
            unique_fold_ids = np.unique(train_fold_ids)
            assert (
                self.repetitions == unique_fold_ids.shape[0]
            ), "self.repetitions != unique_fold_ids.shape[0]"
            iterator = (
                (
                    np.where(train_fold_ids != i)[0].ravel(),
                    np.where(train_fold_ids == i)[0].ravel(),
                )
                for i in unique_fold_ids
            )

        for k, (train_index, val_index) in enumerate(iterator):
            train_index = torch.tensor(train_index).long()
            model_mask = self.output__repetition == k
            model_index = torch.where(model_mask)[0].unsqueeze(1)
            train_mask[train_index, model_index] = True
            model__fold[model_mask] = k

        train_mask = train_mask.bool()
        train_mask = train_mask.to(self.device)
        return train_mask, model__fold

    def fit(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_validation: Tensor,
        y_validation: Tensor,
        x_test: Optional[Tensor],
        y_test: Optional[Tensor],
        current_test_fold: int,
        train_fold_ids: Optional[np.ndarray],  # [fold, indices]
    ):

        x_train, y_train = self.__adjust_data(x_train, y_train)
        x_validation, y_validation = self.__adjust_data(x_validation, y_validation)

        if self.cross_validation and x_validation is not None:
            x_train = torch.cat((x_train, x_validation))
            y_train = torch.cat((y_train, y_validation))

        if y_test is not None:
            x_test, y_test = self.__adjust_data(x_test, y_test)

        nb_labels = len(y_train.unique())
        max_trivial_layers = 1

        (
            hidden_neuron__model_id,
            output__model_id,
            output__architecture_id,
            output__repetition,
            output__activation,
        ) = build_model_ids(
            self.repetitions,
            self.activations,
            self.min_neurons,
            self.max_neurons,
            self.step_neurons,
        )
        # Construct validation
        # unique_fold_ids = np.unique(train_fold_ids)
        # holdout_validation_id = np.random.choice(unique_fold_ids)
        # holdout_mask = train_fold_ids == holdout_validation_id

        # x_validation = x_train[holdout_mask]
        # y_validation = y_train[holdout_mask]

        # x_train = x_train[~holdout_mask]
        # y_train = y_train[~holdout_mask]

        # train_fold_ids = train_fold_ids[~holdout_mask]

        self.train_mask, self.output__kfold = self._generate_model_train_mask(
            len(np.unique(output__model_id)),
            y_train,
            train_fold_ids=train_fold_ids,
        )

        if self.all_data_to_device:
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            if x_validation is not None:
                x_validation = x_validation.to(self.device)
                y_validation = y_validation.to(self.device)
            if y_test is not None:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

        current_train_x = x_train
        current_train_y = y_train

        current_validation_x = x_validation
        current_validation_y = y_validation

        if y_test is not None:
            current_test_x = x_test
            current_test_y = y_test
        else:
            current_test_x = None
            current_test_y = None

        objective = ObjectiveEnum.MINIMIZATION
        global_best_metric = torch.tensor([float("inf")]).to(self.device)

        if self.monitored_metric in ["overall_acc", "matthews_corrcoef"]:
            objective = ObjectiveEnum.MAXIMIZATION
            global_best_metric *= -1

        current_model = nn.Sequential()

        current_patience = 0
        self.current_layer_index = 0

        while current_patience < self.global_patience and (
            self.max_layers is None or self.current_layer_index < self.max_layers
        ):
            self.current_layer_index += 1

            min_neurons = self.min_neurons
            max_neurons = self.max_neurons
            activations = self.activations
            repetitions = self.repetitions
            step_neurons = self.step_neurons

            if self.find_num_neurons_first:
                self.logger.info("Defining architecture first.")
                best_num_neurons, best_activations = self.find_num_neurons(
                    x_train=current_train_x,
                    y_train=current_train_y,
                    x_validation=current_validation_x,
                    y_validation=current_validation_y,
                    x_test=current_test_x,
                    y_test=current_test_y,
                )

                min_neurons = best_num_neurons
                max_neurons = best_num_neurons
                activations = best_activations
                repetitions = self.repetitions_for_best_neuron
                step_neurons = 1

                self.logger.info(
                    f"Best architecture: {best_num_neurons}, activations: {activations}"
                )

            train_dataloader = self._get_dataloader(current_train_x, current_train_y)

            if current_validation_x is not None:
                validation_dataloader = self._get_dataloader(
                    current_validation_x, current_validation_y
                )

            else:
                validation_dataloader = None

            if self.cross_validation:
                validation_dataloader = train_dataloader

            if current_test_x is not None:
                test_dataloader = self._get_dataloader(current_test_x, current_test_y)

            else:
                test_dataloader = None

            (
                hidden_neuron__model_id,
                output__model_id,
                output__architecture_id,
                output__repetition,
                output__activation,
            ) = build_model_ids(
                repetitions,
                activations,
                min_neurons,
                max_neurons,
                step_neurons,
            )
            (
                current_best_mlps,
                current_best_metric,
                model__current_best_metrics,
            ) = self._get_best_mlps(
                hidden_neuron__model_id=hidden_neuron__model_id,
                output__model_id=output__model_id,
                output__architecture_id=output__architecture_id,
                output__repetition=output__repetition,
                activations=activations,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                test_dataloader=test_dataloader,
                from_find_num_neurons=False,
            )

            # current_best_mlp_nb_hidden = current_best_mlps[0].out_features
            # if current_best_mlp_nb_hidden <= nb_labels and len(current_model) > 0:
            #     if max_trivial_layers == 0:
            #         self.logger.info(
            #             f"Current best_mlp hidden neurons: {current_best_mlp_nb_hidden} <= nb_labels {nb_labels}. Stop appending layers."
            #         )
            #         break

            #     max_trivial_layers -= 1

            current_best_mlps_as_mymodels = []
            for best_mlp in current_best_mlps:
                my_model = MyModel(
                    best_mlp,
                    is_ensemble=False,
                    output_confidence=self.output_confidence,
                    min_confidence=self.min_confidence,
                )
                my_model.calculate_activation_statistics(train_dataloader)
                current_best_mlps_as_mymodels.append(my_model)

            current_best_mlps = MyModel(
                current_best_mlps_as_mymodels, is_ensemble=self.is_ensemble
            )
            current_model.add_module(
                name=f"{len(current_model)}", module=current_best_mlps
            )

            # percentage_of_global_best_loss = current_best_validation_loss / (global_best_validation_loss+eps)
            # better_model = percentage_of_global_best_loss < (1-self.min_improvement)
            if self.best_model_sequential is None:
                better_model = (
                    torch.ones_like(current_best_metric)
                    .bool()
                    .unsqueeze(-1)
                    .to(self.device)
                )
                percentage_best_metric = None
            else:
                percentage_best_metric, better_model = helpers.has_improved(
                    current_best_metric.to(self.device),
                    global_best_metric,
                    self.min_improvement,
                    objective,
                    eps,
                )

            if objective == ObjectiveEnum.MINIMIZATION:
                self.logger.info(
                    f"percentage_best_metric({percentage_best_metric}) = current_best_metric({current_best_metric})/global_best_metric({global_best_metric}) < {1-self.min_improvement} (={better_model})."
                )
            else:
                self.logger.info(
                    f"percentage_best_metric({percentage_best_metric}) = current_best_metric({current_best_metric})/global_best_metric({global_best_metric}) > {1+self.min_improvement} (={better_model})."
                )

            if better_model:
                self.best_model_sequential = deepcopy(current_model)
                self.logger.info(
                    f"Improved {self.monitored_metric} from {global_best_metric} to {current_best_metric}. Setting current_patience=0. Current best model with {len(self.best_model_sequential)-1} layers: ({self.best_model_sequential})."
                )
                global_best_metric = current_best_metric.to(self.device)
                current_patience = 0
            else:
                current_patience += 1
                self.logger.info(
                    f"No improvement. current_patience={current_patience}."
                )

            if objective == ObjectiveEnum.MINIMIZATION:
                if current_best_metric < eps:
                    self.logger.info(
                        f"current_best_metric ({current_best_metric}) < eps ({eps}). Stopping fit."
                    )
                    break

            if self.stack_hidden_layers:
                # current_best_mlps = current_best_mlps[:-1]
                for cbmlp in current_best_mlps:
                    cbmlp.module.out_layer = None

                # current_model[-1] = current_model[-1][:-1]
                for cm in current_model[-1]:
                    cm.module.out_layer = None

            # if len(current_model) < self.max_layers:
            (
                current_train_x,
                current_validation_x,
                current_test_x,
            ) = self._apply_forward_transform_data(
                x_train,
                x_validation,
                x_test,
                current_train_x,
                current_validation_x,
                current_test_x,
                current_best_mlps,
            )
        self.logger.info(f"final_model: {self.best_model_sequential}.")

    def get_best_model_arch(self):
        if self.is_ensemble:
            return "ensemble"
        else:
            arch = [
                self.best_model_sequential[0].module[0].module.hidden_layer.in_features
            ]
            for model in self.best_model_sequential:
                mlps = model.module
                for mlp in mlps:
                    # arch.append(mlp.module.out_layer.in_features)
                    arch.append(mlp.module.hidden_layer.out_features)

            arch.append(mlp.module.out_layer.out_features)
            return arch

    def predict(self, x: Tensor):
        x, _ = self.__adjust_data(x, None)
        x = x.to(self.device)
        h = x
        total_layers = len(self.best_model_sequential)
        for i, layer in enumerate(self.best_model_sequential):
            h = layer(h)

            if i < total_layers - 1:
                h = self._transform_data(x, h)

        return h
