import pathlib
import sys
from typing import List
from dataclasses import dataclass
import tomllib


@dataclass
class ProjectConfig:
    name: str
    experiment_name: str


@dataclass
class DataConfig:
    raw_dir: str
    split_dir: str
    batch_size: int
    num_workers: int
    image_size: List[int]
    seed: int


@dataclass
class ModelConfig:
    num_layers: int
    dropout: float
    dim_ratio: int
    num_heads: int
    patch_size: int
    n_embd: int
    max_seq_len: int


@dataclass
class TrainingConfig:
    lr: float
    weight_decay: float
    warmup_steps: int
    betas: List[float]
    grad_clip: float
    val_every_steps: int
    vis_every_steps: int
    scheduler: str
    max_steps: int


@dataclass
class LoggingConfig:
    use_wandb: bool
    project_name: str
    assets_dir: str


@dataclass
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str = "configs/base.toml") -> "Config":
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found at {path_obj.resolve()}")

        with open(path_obj, "rb") as f:
            data = tomllib.load(f)

        return cls(
            project=ProjectConfig(**data["project"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
            logging=LoggingConfig(**data["logging"]),
        )
