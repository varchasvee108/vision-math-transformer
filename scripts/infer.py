import torch
from PIL import Image
from core.config import Config
from core.factory import build_inference_components


def infer(
    image_path: str,
    config_path: str = "configs/base.toml",
    model_weights="experiments/base/latest.pth",
):
    config = Config.load(config_path)
    model, processor, device = build_inference_components(config=config)

    ckpt = torch.load(model_weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img = Image.open(image_path).convert("L")
    pixel_values = processor.process_image(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        generated_id = torch.tensor([[processor.sos_id]], device=device)

        for _ in range(config.model.max_seq_len):
            logits = model(pixel_values, generated_id)

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            generated_id = torch.cat([generated_id, next_token_id], dim=-1)

            if next_token_id.item() == processor.eos_id:
                break

    prediction = processor.decode(generated_id[0])
    return prediction
