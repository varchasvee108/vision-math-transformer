import torch
from core.config import Config
from models.layers import PatchEmbedding, EncoderBlock, DecoderBlock


def test_patch_embedding():
    config = Config.load("configs/base.toml")
    layer = PatchEmbedding(config=config)

    B = 2
    H, W = config.data.image_size
    x = torch.randn(B, 1, H, W)

    y = layer(x)

    grid = H // config.model.patch_size

    expected_tokens = grid * grid
    assert y.shape == (B, expected_tokens, config.model.n_embd)
    assert torch.isfinite(y).all()
