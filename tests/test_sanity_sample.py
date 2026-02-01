import torch
from PIL import Image
from core.config import Config
from core.factory import build_inference_components


def test_sanity_sample():
    config = Config.load("configs/base.toml")
    model, processor, device = build_inference_components(config=config)
