import os
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


def already_executed(cfg, run_name, logger, wandb_mode):
    if wandb_mode == "online":
        wandb_api = wandb.Api()
        try:
            runs_in_wandb = [
                r
                for r in wandb_api.runs(cfg.training.project_name)
                if r.state in ["finished", "running"]
                if run_name == r.name
            ]

            if len(runs_in_wandb) > 0:
                logger.info(f"Run {run_name} already executed.")
                return True
        except Exception as e:
            logger.warning(e)

    return False


@hydra.main(config_path="conf", config_name="config")
def main(cfg: AutoConstructiveConfig) -> None:
    run_name = f"{cfg.training.dataset}_{cfg.training.experiment_num}"

    wandb_mode = os.environ["WANDB_MODE"]

    if already_executed(cfg, run_name, logger, wandb_mode):
        return

    print(OmegaConf.to_yaml(cfg))
    dl = Dataloader(
        dataset_name=cfg.training.dataset,
        n_splits=cfg.training.n_splits,
        feature_range=(0, 1),
        random_state=cfg.training.experiment_num,
        data_home=cfg.training.data_home,
        log=logger,
        one_hot_encode_output=False,
    )

    if cfg.training.distance_name is None:
        splits = dl.get_splits_iter(
            validation_rate_from_train=cfg.training.validation_rate_from_train,
        )
    else:

        splits = dl.get_splits_iter_regions(
            validation_rate_from_train=cfg.training.validation_rate_from_train,
            distance_name="correlation",
        )

    next_kfold = cfg.training.experiment_num % cfg.training.n_splits
    for i, data in enumerate(tqdm(splits, desc="KFold", total=cfg.training.n_splits)):
        if i < next_kfold:
            continue

        with wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.training.project_name,
            mode=wandb_mode,
            reinit=True,
            name=run_name,
        ):
            # config = wandb.config
            logger.info(f"wandb.run.name: {wandb.run.name}")

            try:
                return run_single_experiment(data, cfg, logger)
            except Exception as e:
                print(f"print {e}")
                logger.info(f"logger-info {e}")
                logger.error(f"logger-error {e}")
                logger.error(traceback.format_exc())
                raise e
            logger.info("FINISHED!!!")


def run_single_experiment(data: Dict, cfg: AutoConstructiveConfig, logger: Any):
    random_state = cfg.training.experiment_num
    auto_constructive = AutoConstructiveModel(
        all_data_to_device=cfg.model.all_data_to_device,
        loss_function=resolve_loss_function(cfg.model.loss_function),
        optimizer_name=cfg.model.optimizer_name,
        learning_rate=cfg.model.learning_rate,
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        drop_samples=cfg.training.drop_samples,
        input_perturbation=cfg.training.input_perturbation_strategy,
        regularization_gamma=cfg.training.regularization_gamma,
        monitored_metric=cfg.training.monitored_metric,
        find_num_neurons_first=cfg.training.find_num_neurons_first,
        mcdm_weights=cfg.training.mcdm_weights,
        num_workers=cfg.model.num_workers,
        repetitions=cfg.model.repetitions,
        repetitions_for_best_neuron=cfg.model.repetitions_for_best_neuron,
        activations=resolve_activations(cfg.model.activations),
        topk=cfg.model.topk,
        output_confidence=cfg.model.output_confidence,
        min_confidence=cfg.model.min_confidence,
        min_neurons=cfg.model.min_neurons,
        max_neurons=cfg.model.max_neurons,
        max_layers=cfg.model.max_layers,
        stack_hidden_layers=cfg.model.stack_hidden_layers,
        step_neurons=cfg.model.step_neurons,
        local_patience=cfg.model.local_patience,
        global_patience=cfg.model.global_patience,
        transform_data_strategy=cfg.model.transform_data_strategy,
        loss_rel_tol=cfg.model.loss_rel_tol,
        min_improvement=cfg.model.min_improvement,
        device=cfg.model.device,
        random_state=random_state,
        logger=logger,
        debug_test=cfg.training.debug_test,
        reset_exhausted_models=cfg.training.reset_exhausted_models,
    )

    x_train = data["train"]["data"]
    y_train = data["train"]["target"]
    y_train = y_train.squeeze()
    logger.info(f"Train set: {x_train.shape}")

    if "val" in data.keys():
        x_val = data["val"]["data"]
        y_val = data["val"]["target"]
        y_val = y_val.squeeze()
        logger.info(f"Validation set: {x_val.shape}")

    else:
        x_val = None
        y_val = None

    x_test = data["test"]["data"]
    y_test = data["test"]["target"]
    y_test = y_test.squeeze()
    logger.info(f"Test set: {x_test.shape}")
    if cfg.training.debug_test:
        x_test_debug = x_test
        y_test_debug = y_test

    else:
        x_test_debug = None
        y_test_debug = None

    start = perf_counter()

    # profiler = Profiler()
    # profiler.start()
    auto_constructive.fit(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_val,
        y_validation=y_val,
        x_test=x_test_debug,
        y_test=y_test_debug,
    )

    # profiler.stop()
    # profiler.open_in_browser()
    end = perf_counter()
    logger.info(f"auto_constructive.fit took: {end-start} ")

    if False:
        test_sequential(
            torch.tensor(x_train),
            torch.tensor(y_train),
            torch.tensor(x_val),
            torch.tensor(y_val),
            torch.tensor(x_test),
            torch.tensor(y_test),
            auto_constructive,
            cfg,
        )

    wandb.run.summary.update(
        {
            "training_time": end - start,
            "architecture": auto_constructive.get_best_model_arch(),
            "num_trained_mlps": auto_constructive.num_trained_mlps,
        }
    )

    auto_constructive.eval()
    with torch.no_grad():
        log_results(
            {"autoconstructive": auto_constructive.predict(x_train).cpu().numpy()},
            y_train,
            "train",
        )
        if x_val is not None:
            log_results(
                {"autoconstructive": auto_constructive.predict(x_val).cpu().numpy()},
                y_val,
                "val",
            )
        if x_test is not None:
            log_results(
                {"autoconstructive": auto_constructive.predict(x_test).cpu().numpy()},
                y_test,
                "test",
            )

    evaluate_classic_models(
        x_train, y_train, x_val, y_val, x_test, y_test, random_state
    )


def evaluate_classic_models(
    x_train, y_train, x_validation, y_validation, x_test, y_test, random_state
):
    models = {
        "1-nn": sklearn.neighbors.KNeighborsClassifier(n_neighbors=1),
        "3-nn": sklearn.neighbors.KNeighborsClassifier(n_neighbors=3),
        "svm": sklearn.svm.SVC(probability=True, random_state=random_state),
        "xgboost": xgboost.XGBClassifier(random_state=random_state),
        "rf": sklearn.ensemble.RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }
    for model_name, model in models.items():
        # Reproducibility
        np.random.seed(random_state)

        #  __import__('pudb').set_trace()
        # y_train = y_train.argmax(1)

        model.fit(x_train, y_train)

        model.ohe = sklearn.preprocessing.OneHotEncoder().fit(y_train[:, None])
        logits_val = None
        logits_train = model.predict_proba(x_train)
        # logits_train = model.ohe.transform(logits_train[:, None]).toarray()
        if x_validation is not None:
            logits_val = model.predict_proba(x_validation)
            # logits_val = model.ohe.transform(logits_val[:, None]).toarray()

        # logits_test = model.ohe.transform(logits_test[:, None]).toarray()

        log_results({model_name: logits_train}, y_train, "train")
        log_results({model_name: logits_val}, y_validation, "val")
        if x_test is not None:
            logits_test = model.predict_proba(x_test)
            log_results({model_name: logits_test}, y_test, "test")


def log_results(logits, y, metric_prefix, ignore_wandb=False):
    metrics = assess_model(logits, y, metric_prefix)
    if not ignore_wandb:
        print(metrics)
        for train_metric in metrics:
            wandb.run.summary.update(train_metric)

    return metrics


def test_sequential(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    autoconstructive,
    cfg: AutoConstructiveConfig,
):
    if cfg.model.all_data_to_device:
        x_train = x_train.to(cfg.model.device)
        y_train = y_train.to(cfg.model.device)
        x_validation = x_validation.to(cfg.model.device)
        y_validation = y_validation.to(cfg.model.device)
        x_test = x_test.to(cfg.model.device)
        y_test = y_test.to(cfg.model.device)

    train_dataset = TensorDataset(x_train, y_train)
    validation_dataset = TensorDataset(x_validation, y_validation)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=cfg.training.batch_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

    (
        hidden_neuron__model_id,
        output__model_id,
        output__architecture_id,
        output__repetition,
    ) = build_model_ids(
        cfg.model.repetitions,
        resolve_activations(cfg.model.activations),
        cfg.model.min_neurons,
        cfg.model.max_neurons,
        cfg.model.step_neurons,
    )

    in_features = train_dataloader.dataset[0][0].shape[0]
    out_features = len(train_dataloader.dataset.tensors[-1].unique())

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
    )

    start = perf_counter()
    num_models = pmlps.num_unique_models
    i = 0
    for model_id in tqdm(range(num_models)):
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
            for (x, y) in train_dataloader:
                optimizer.zero_grad()
                p = model(x)
                l = loss_function(p, y)
                l.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for (x, y) in validation_dataloader:
                    p = model(x)
                    loss_function(p, y)

                for (x, y) in test_dataloader:
                    p = model(x)
                    loss_function(p, y)
    end = perf_counter()
    logger.info(f"Sequential took: {end-start} to train {i} models")


if __name__ == "__main__":
    main()