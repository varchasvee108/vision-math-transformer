import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from tqdm import tqdm
from pathlib import Path
from core.factory import TrainingComponents
from core.config import Config


class Trainer:
    def __init__(self, config: Config, components: TrainingComponents):
        self.config = config
        self.model = components.model
        self.optimizer = components.optimizer
        self.scheduler = components.lr_scheduler
        self.train_dataloader = components.train_dataloader
        self.val_dataloader = components.val_dataloader
        self.processor = components.processor
        self.device = components.device

        self.exp_dir = Path("experiments") / config.project.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        if self.config.logging.use_wandb:
            wandb.init(
                project=config.logging.project_name,
                name=config.project.experiment_name,
                config=config,
            )

    def train_step(self, batch):
        self.model.train()

        assert batch["image"].ndim == 4

        images = batch["image"].to(self.device)
        labels = batch["target"].to(self.device)

        decoder_inputs = labels[:, :-1]
        decoder_targets = labels[:, 1:]

        logits = self.model(images, decoder_inputs)

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), decoder_targets.view(-1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
