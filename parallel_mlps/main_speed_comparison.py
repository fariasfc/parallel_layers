import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torch.optim.sgd import SGD
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

# from pyinstrument import Profiler
from typing import Any, Dict
from autoconstructive.model.parallel_mlp import ParallelMLPs, build_model_ids
from conf.config import (
    AutoConstructiveConfig,
    resolve_activations,
    resolve_loss_function,
    resolve_optimizer_type,
)
from time import perf_counter
import numpy as np
import xgboost
import sklearn
import sklearn.ensemble
from tqdm import tqdm
import traceback
from experiment_utils import assess_model
from omegaconf import DictConfig
import hydra
from omegaconf import OmegaConf
from autoconstructive.model.autoconstructive_model import AutoConstructiveModel
from data.dataloader import Dataloader
import torch
from torch import nn
from torch import optim
import wandb

import logging

logger = logging.getLogger()

def test_sequential(
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    pmlps,
    cfg: AutoConstructiveConfig,
    with_gradients = True,
):
    in_features = pmlps.in_features
    out_features = pmlps.out_features

    num_models = pmlps.num_unique_models
    i = 0
    sequential_etas = []
    # for model_id in tqdm(range(num_models)):
    for model_id in range(num_models):
        i += 1
        num_hidden = pmlps.model_id__num_hidden_neurons[model_id]
        activation = pmlps.get_activation_from_model_id(model_id)
        model = nn.Sequential(
            nn.Linear(in_features, num_hidden),
            activation,
            nn.Linear(num_hidden, out_features),
        )
        model = model.to(cfg.model.device)
        optimizer = SGD(params=model.parameters(), lr=cfg.model.learning_rate)
        loss_function = nn.CrossEntropyLoss()


        for epoch in range(cfg.training.num_epochs):
            model.train()
            if with_gradients:
                start = perf_counter()
                for (x, y) in train_dataloader:
                    optimizer.zero_grad()
                    p = model(x)
                    l = loss_function(p, y)
                    l.backward()
                    optimizer.step()
                end = perf_counter()
            else:
                model.eval()
                with torch.no_grad():
                    start = perf_counter()
                    for (x, y) in train_dataloader:
                        p = model(x)
                        loss_function(p, y)
                    end = perf_counter()

            if epoch >= 2:
                sequential_etas.append(end-start)

    average_eta = np.sum(sequential_etas)/(cfg.training.num_epochs-2)
    logger.info(f"Sequential: Each epoch took on average ({len(sequential_etas)} recorded epochs*models) {average_eta} per epoch to train {i} models for {cfg.training.num_epochs-2} epochs")
    return average_eta

@hydra.main(config_path="conf", config_name="config")
def main_sequential_parallel(cfg: AutoConstructiveConfig) -> None:
    num_features_list = [5, 10, 50, 100]
    num_samples_list = [100, 1000, 10000]
    num_features_list = [100, 50, 10, 5]
    num_samples_list = [10000, 1000, 100]
    # num_features_list = [5, 10]
    # num_samples_list = [10, 100]
    num_epochs_list = [10]
    d = []
    with_gradients = True
    for num_features in num_features_list:
        for num_samples in num_samples_list:
            x_train = torch.randn(num_samples, num_features)
            y_train = torch.randint(low=0, high=1, size=(num_samples,))
            x_validation = torch.randn(64, num_features)
            y_validation = torch.randint(low=0, high=1, size=(64,))
            x_test = torch.randn(64, num_features)
            y_test = torch.randint(low=0, high=1, size=(64,))

            if cfg.model.all_data_to_device:
                x_train = x_train.to(cfg.model.device)
                y_train = y_train.to(cfg.model.device)
                if x_validation is not None:
                    x_validation = x_validation.to(cfg.model.device)
                    y_validation = y_validation.to(cfg.model.device)
                if x_test is not None:
                    x_test = x_test.to(cfg.model.device)
                    y_test = y_test.to(cfg.model.device)

            for num_epochs in num_epochs_list:
                print(f"Simulations with num_features={num_features}, num_samples={num_samples}, num_epochs={num_epochs}")
                cfg.training.num_epochs = num_epochs+2

                (
                    hidden_neuron__model_id,
                    output__model_id,
                    output__architecture_id,
                    output__repetition,
                    output__activation,
                ) = build_model_ids(
                    cfg.model.repetitions,
                    resolve_activations(cfg.model.activations),
                    cfg.model.min_neurons,
                    cfg.model.max_neurons,
                    cfg.model.step_neurons,
                )

                in_features = x_train.shape[1]
                out_features = len(y_train.unique())

                pmlps = ParallelMLPs(
                    in_features,
                    out_features,
                    hidden_neuron__model_id,
                    output__model_id,
                    output__architecture_id,
                    output__repetition,
                    None,
                    None,
                    resolve_activations(cfg.model.activations),
                    device=cfg.model.device
                ).to(cfg.model.device)
                optimizer = SGD(pmlps.parameters(),lr=cfg.model.learning_rate)
                loss_function=resolve_loss_function(cfg.model.loss_function)
                loss_function.reduction = "none"
                dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.training.batch_size)
                parallel_etas = []
                for epoch in range(cfg.training.num_epochs):
                    pmlps.train()
                    if with_gradients:
                        start = perf_counter()
                        for x, y in dataloader:
                            #x = x.to(cfg.model.device)
                            #y = y.to(cfg.model.device)

                            optimizer.zero_grad()

                            outputs = pmlps(x)  # [batch_size, num_models, out_features]

                            individual_losses = pmlps.calculate_loss(
                                loss_func=loss_function, preds=outputs, target=y
                            )

                            loss = individual_losses.mean(
                                0
                            )  # [batch_size, num_models] -> [num_models]

                            loss = loss.sum()  #  [num_models]-> []

                            loss.backward()
                            optimizer.step()
                        end = perf_counter()
                    else:
                        pmlps.eval()
                        with torch.no_grad():
                            start = perf_counter()
                            for x, y in dataloader:
                                outputs = pmlps(x)  # [batch_size, num_models, out_features]

                                individual_losses = pmlps.calculate_loss(
                                    loss_func=loss_function, preds=outputs, target=y
                                )

                                loss = individual_losses.mean(
                                    0
                                )  # [batch_size, num_models] -> [num_models]

                                loss = loss.sum()  #  [num_models]-> []
                            end = perf_counter()

                    if epoch >= 2:
                        parallel_etas.append(end-start)

                average_eta_parallel = np.sum(parallel_etas)/(cfg.training.num_epochs-2)
                logger.info(f"Parallel: Each epoch took on average  ({len(parallel_etas)} recorded epochs) {average_eta_parallel} to train {pmlps.num_unique_models} models for {cfg.training.num_epochs-2} epochs")

                average_eta_sequential = test_sequential(train_dataloader=dataloader, validation_dataloader=None, test_dataloader=None, pmlps=pmlps, cfg=cfg, with_gradients=with_gradients)
                d.append({
                    "num_samples": num_samples,
                    "num_features": num_features,
                    "min_neurons": cfg.model.min_neurons,
                    "max_neurons": cfg.model.max_neurons,
                    "epochs": cfg.training.num_epochs,
                    "num_models": pmlps.num_unique_models,
                    "activation_functions": cfg.model.activations,
                    "repetitions": cfg.model.repetitions,
                    "sequential": average_eta_sequential,
                    "parallel": average_eta_parallel,
                    "device": cfg.model.device,
                })
            del pmlps
            pmlps = None
    df = pd.DataFrame(d)
    df["parallel/sequential"] = df["parallel"]/df["sequential"]
    df_path = Path(f"times_{cfg.model.device}_with_gradients_{with_gradients}.csv")
    print(f"Saving df to {df_path.absolute()}")
    df.to_csv(df_path)
    print(df)



if __name__ == "__main__":
    main_sequential_parallel()

# python main_speed_comparison.py model.min_neurons=1 model.max_neurons=10 model.activation_functions=[identity, relu] model.device=cpu model.repetitions=5 
#python main_speed_comparison.py model.min_neurons=1 model.max_neurons=10 model.activations=\[identity,relu\] model.device=cpu model.repetitions=5  
