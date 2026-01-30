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


def encoder_layer_test():
    config = Config.load("configs/base.toml")
    layer = EncoderBlock(config)

    B = 12
    N = 20

    x = torch.randn(B, N, config.model.n_embd)

    y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def decoder_layer_test():
    config = Config.load("configs/base.toml")
    layer = DecoderBlock(config=config)

    B = 18
    T = 4
    N = 16

    decoder_x = torch.randn(B, T, config.model.n_embd)
    encoder_x = torch.randn(B, N, config.model.n_embd)

    y = layer(decoder_x, encoder_x)

    assert y.shape == decoder_x.shape
    assert torch.isfinite(y).all()
