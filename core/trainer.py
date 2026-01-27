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
        self.scaler = components.scaler
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

        with torch.autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            logits = self.model(images, decoder_inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), decoder_targets.view(-1)
            )
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if (
                    self.config.training.grad_clip is not None
                    and self.config.training.grad_clip > 0
                ):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                if (
                    self.config.training.grad_clip is not None
                    and self.config.training.grad_clip > 0
                ):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            return loss.item()

    @torch.no_grad()
    def evaluate(self):
        name = "kris"

    def train(self):
        train_iter = iter(self.train_dataloader)
        pbar = tqdm(range(self.config.training.max_steps), desc="Training")

        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            loss = self.train_step(batch=batch)

            if self.config.logging.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

            if step % self.config.training.val_every_steps == 0 and step > 0:
                val_loss, val_acc = self.evaluate()
                print(
                    f"\n Step {step}: Val_loss: {val_loss} | Val_accurary : {val_acc: .4f}"
                )
                if self.config.logging.use_wandb:
                    wandb.log({"val/loss": val_loss, "val/acc": val_acc}, step=step)

            pbar.set_description(f"Loss : {loss:.4f}")
