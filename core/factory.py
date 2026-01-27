from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core.config import Config
from models.model import VisionMathTransformer
from data.dataloader import create_dataloader
from data.processor import VisionProcessor
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler


@dataclass
class TrainingComponents:
    model: VisionMathTransformer
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    processor: VisionProcessor
    device: torch.device


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_processor(config: Config):
    return VisionProcessor(
        image_size=tuple(config.data.image_size), vocab=config.data.vocab
    )


def create_model(config: Config, device: torch.device):
    model = VisionMathTransformer(config).to(device)
    return model


def create_optim_and_scheduler(
    config: Config, model: VisionMathTransformer
) -> Tuple[torch.optim.Optimizer, LRScheduler]:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("bias"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.lr,
        betas=config.training.betas,
    )

    scheduler = get_scheduler(
        config.training.scheduler,
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.num_training_steps,
    )

    return optimizer, scheduler


def build_training_components(config: Config) -> TrainingComponents:
    device = get_device()
    processor = create_processor(config)
    model = create_model(config, device)
    train_dataloader, val_dataloader = create_dataloader(config, processor=processor)
