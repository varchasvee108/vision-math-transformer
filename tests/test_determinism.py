import torch
from core.config import Config
from models.model import VisionMathTransformer
from data.processor import VisionProcessor


def test_deterministic_forward():
    config = Config.load("configs/base.toml")
    processor = VisionProcessor(
        image_size=config.data.image_size, vocab=config.data.vocab
    )
    model = VisionMathTransformer(config, pad_id=processor.pad_id)

    model.eval()

    B = 2
    H, W = config.data.image_size
    T = 4
    vocab_size = len(config.data.vocab)

    image_tensor = torch.randn(B, 1, H, W)
    decoder_input_ids = torch.randint(0, vocab_size, (B, T))

    torch.manual_seed(1008)
    with torch.inference_mode():
        y1 = model(image_tensor, decoder_input_ids)

    torch.manual_seed(1008)
    with torch.inference_mode():
        y2 = model(image_tensor, decoder_input_ids)

    assert torch.allclose(y1, y2)
