import torch
from core.config import Config
from models.model import VisionMathTransformer


def test_model_output_shape(path="configs/base.toml"):
    config = Config.load(path)
    model = VisionMathTransformer(config)
    B = 2
    H, W = config.data.image_size
    T = 4
    vocab_size = len(config.data.vocab)

    image_tensor = torch.randn(B, 1, H, W)
    decoder_input_ids = torch.randint(0, vocab_size, (B, T))
    model.eval()

    with torch.inference_mode():
        logits = model(image_tensor, decoder_input_ids)
    assert logits.shape == (B, T, vocab_size)
    assert torch.isfinite(logits).all()
