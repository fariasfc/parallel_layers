from conf.config import AutoConstructiveConfig
import hydra
from omegaconf import OmegaConf
from model.autoconstructive import AutoConstructive
import torch
from torch import nn
from torch import optim


@hydra.main(config_path="conf", config_name="config")
def main(cfg: AutoConstructiveConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    auto_constructive = AutoConstructive(
        all_data_to_device=True,
        loss_function=nn.CrossEntropyLoss(),
        optimizer_cls=optim.Adam,
        learning_rate=3e-4,
        num_epochs=5,
        batch_size=32,
        num_workers=6,
        repetitions=2,
        activations=[nn.ReLU(), nn.Sigmoid()],
        min_neurons=1,
        max_neurons=3,
        step_neurons=1,
        device="cuda",
    )

    auto_constructive.fit(
        train_x=torch.randn(
            (100, 5),
        ),
        train_y=torch.randint(0, 2, size=(100,)),
        validation_x=torch.randn(
            (50, 5),
        ),
        validation_y=torch.randint(0, 2, size=(50,)),
    )


if __name__ == "__main__":
    main()