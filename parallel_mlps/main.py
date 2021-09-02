import os
from typing import Any, Dict
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

def already_executed(cfg, run_name, logger):
    wandb_api = wandb.Api()
    runs_in_wandb = [
        r
        for r in wandb_api.runs(cfg.training.project_name)
        if r.state in ["finished", "running"]
        if run_name == r.name
    ]

    if len(runs_in_wandb) > 0:
        logger.info(f"Run {run_name} already executed.")
        return True

    else:
        return False

@hydra.main(config_path="conf", config_name="config")
def main(cfg: AutoConstructiveConfig) -> None:
    run_name = f"{cfg.training.dataset}_{cfg.training.experiment_num}"

    if already_executed(cfg, run_name, logger):
        return

    print(OmegaConf.to_yaml(cfg))
    wandb_mode = os.environ["WANDB_MODE"]
    dl = Dataloader(
        dataset_name=cfg.training.dataset,
        n_splits=cfg.training.n_splits,
        feature_range=(0, 1),
        random_state=cfg.training.experiment_num,
        data_home=cfg.training.data_home,
        log=logger,
        one_hot_encode_output=False,
    )

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
        optimizer_type=resolve_optimizer_type(cfg.model.optimizer_cls),
        learning_rate=cfg.model.learning_rate,
        num_epochs=cfg.model.num_epochs,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        repetitions=cfg.model.repetitions,
        activations=resolve_activations(cfg.model.activations),
        min_neurons=cfg.model.min_neurons,
        max_neurons=cfg.model.max_neurons,
        step_neurons=cfg.model.step_neurons,
        local_patience=cfg.model.local_patience,
        global_patience=cfg.model.global_patience,
        transform_data_strategy=cfg.model.transform_data_strategy,
        loss_rel_tol=cfg.model.loss_rel_tol,
        min_improvement=cfg.model.min_improvement,
        device=cfg.model.device,
        random_state=random_state,
        logger=logger,
    )

    x_train = data["train"]["data"]
    y_train = data["train"]["target"]
    logger.info(f"Train set: {x_train.shape}")

    if "val" in data.keys():
        x_val = data["val"]["data"]
        y_val = data["val"]["target"]
        logger.info(f"Validation set: {x_val.shape}")

    else:
        x_val = None
        y_val = None

    x_test = data["test"]["data"]
    y_test = data["test"]["target"]

    logger.info(f"Test set: {x_test.shape}")

    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    start = perf_counter()
    auto_constructive.fit(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_val,
        y_validation=y_val,
    )
    end = perf_counter()

    wandb.run.summary.update(
        {
            "training_time": end - start,
            "architecture": auto_constructive.get_best_model_arch(),
            "num_trained_mlps": auto_constructive.num_trained_mlps
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

        logits_test = model.predict_proba(x_test)
        # logits_test = model.ohe.transform(logits_test[:, None]).toarray()

        log_results({model_name: logits_train}, y_train, "train")
        log_results({model_name: logits_val}, y_validation, "val")
        log_results({model_name: logits_test}, y_test, "test")


def log_results(logits, y, metric_prefix):
    metrics = assess_model(logits, y, metric_prefix)
    print(metrics)
    for train_metric in metrics:
        wandb.run.summary.update(train_metric)


if __name__ == "__main__":
    main()