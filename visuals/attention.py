import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import imageio
import plotly.graph_objects as go
import plotly.express as px
from core.config import Config
from core.factory import build_inference_components


def attention_to_image(attn_1d, grid_size, image_size):
    H, W = image_size
    attn_2d = attn_1d.view(grid_size, grid_size)
    attn_2d = attn_2d.unsqueeze(0).unsqueeze(0)
    attn_upsampled = F.interpolate(
        attn_2d, size=(H, W), mode="bilinear", align_corners=False
    )
    attn_upsampled = attn_upsampled.squeeze().cpu().numpy()
    attn_upsampled -= attn_upsampled.min()
    attn_upsampled /= attn_upsampled.max() + 1e-8
    return attn_upsampled


def save_attention_gif(
    image, attention_maps, tokens, out_path="cross_attention.gif", fps=2
):
    frames = []
    image_np = np.array(image)

    for i, attn in enumerate(attention_maps):
        fig_left = px.imshow(image, color_continuous_scale="gray")
        fig_left.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig_left.write_image("_left.png", scale=2)

        fig_right = px.imshow(image, color_continuous_scale="gray")
        fig_right.add_trace(
            go.Heatmap(
                z=attn,
                colorscale="jet",
                opacity=0.5,
                showscale=False,
            )
        )
        fig_right.update_layout(
            title=f"Step {i+1} | Token: {tokens[i]}",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig_right.write_image("_right.png", scale=2)

        left = imageio.imread("_left.png")
        right = imageio.imread("_right.png")
        combined = np.concatenate([left, right], axis=1)

        frames.append(combined)

        os.remove("_left.png")
        os.remove("_right.png")

    imageio.mimsave(out_path, frames, fps=fps)


def plot_3d_attention(attention_map, title="3D Cross-Attention"):
    fig = go.Figure(
        data=[
            go.Surface(
                z=attention_map,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Attention"),
        ),
        height=600,
        width=600,
    )
    fig.show()


def run_attention_visualization(
    image_path: str,
    model_weights: str = "experiments/base/latest.pth",
):
    config = Config.load()
    model, processor, device = build_inference_components(config)

    checkpoint = torch.load(model_weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = Image.open(image_path).convert("L")
    pixel_values = processor.process_image(img).unsqueeze(0).to(device)

    generated_ids = torch.tensor([[processor.sos_id]], device=device)
    attention_frames = []
    decoded_tokens = []

    with torch.inference_mode():
        for _ in range(config.model.max_seq_len):
            logits, attn = model(pixel_values, generated_ids, return_attn=True)
            next_token = logits[:, -1].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token[:, None]], dim=1)

            if next_token.item() == processor.eos_id:
                break

            decoded_tokens.append(processor.itos[next_token.item()])

            cross_attn = attn["decoder_cross_attn"]
            cross_attn = cross_attn[0].mean(dim=0)
            attn_token = cross_attn[-1]

            grid_size = config.data.image_size[0] // config.model.patch_size
            attention_map = attention_to_image(
                attn_token,
                grid_size=grid_size,
                image_size=tuple(config.data.image_size),
            )

            attention_frames.append(attention_map)

    save_attention_gif(
        img,
        attention_frames,
        decoded_tokens,
        out_path="cross_attention.gif",
        fps=2,
    )

    plot_3d_attention(attention_frames[-1], title="Final Token Cross-Attention (3D)")


if __name__ == "__main__":
    run_attention_visualization("data/images/test.png")
