import torch
from PIL import Image
from core.config import Config
from core.factory import build_inference_components
import pytest


@pytest.mark.integration
def test_sanity_sample():
    config = Config.load("configs/base.toml")

    model, processor, device = build_inference_components(config=config)
    model.eval()

    img = Image.new("L", tuple(config.data.image_size), color=255)
    pixel_values = processor.process_image(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        generated_ids = torch.tensor([[processor.sos_id]], device=device)

        for _ in range(config.model.max_seq_len):
            logits = model(pixel_values, generated_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id.item() == processor.eos_id:
                break

    output = processor.decode(generated_ids[0])

    assert isinstance(output, str)
    assert len(output) <= config.model.max_seq_len
    assert all(ch.isdigit() for ch in output) or output == ""
