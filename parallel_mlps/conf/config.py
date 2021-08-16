from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class Training:
    dataset: str


@dataclass
class AutoConstructiveConfig:
    training: Training


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=AutoConstructiveConfig)
# cs.store(name="training", node=Training)
