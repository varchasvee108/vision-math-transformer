from torch.utils.data import DataLoader
from core.config import Config
from datasets.dataset import VisionMathDataset
from data.processor import VisionProcessor


def create_dataloader(config: Config, processor: VisionProcessor):
    train_dataset = VisionMathDataset(config=config, processor=processor, split="train")
    train_dataloder = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_dataset = VisionMathDataset(config=config, processor=processor, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    return train_dataloder, val_dataloader
