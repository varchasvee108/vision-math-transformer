import torch
from core.config import Config
from models.model import VisionMathTransformer
from data.processor import VisionProcessor


def test_model_output_shape(
    path="configs/base.toml",
):
    config = Config.load(path)
    processor = VisionProcessor(
        image_size=config.data.image_size, vocab=config.data.vocab
    )
    model = VisionMathTransformer(config, pad_id=processor.pad_id)
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
