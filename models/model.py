import torch
import torch.nn as nn
from core.config import Config
from models.layers import DecoderBlock, EncoderBlock, PatchEmbedding, MLP
from data.processor import VisionProcessor


class VisionMathTransformer(nn.Module):
    def __init__(self, config: Config, pad_id: int):
        super().__init__()
        self.config = config
        self.n_embd = config.model.n_embd
        self.pad_id = pad_id
        self.patch_embd = PatchEmbedding(config)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.model.num_layers)]
        )
        self.encoder_ln = nn.LayerNorm(self.n_embd)
        self.decoder_embd = nn.Embedding(len(config.data.vocab), self.n_embd)
        self.decoder_pos_embd = nn.Embedding(config.model.max_seq_len, self.n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.model.num_layers)]
        )
        self.decoder_ln = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, len(config.data.vocab))

    def generate_causal_mask(self, T, device):
        return torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, image_tensor, decoder_input_ids, return_attn=False):

        assert image_tensor.ndim == 4
        assert decoder_input_ids.ndim == 2
        assert image_tensor.device == decoder_input_ids.device
        pad_mask = decoder_input_ids == self.pad_id

        B_img = image_tensor.shape[0]
        B_dec, T_dec = decoder_input_ids.shape
        assert B_img == B_dec
        assert T_dec <= self.config.model.max_seq_len

        x = self.patch_embd(image_tensor)

        for block in self.encoder_blocks[:-1]:
            x = block(x)

        if return_attn:
            x, encoder_attn = self.encoder_blocks[-1](x, return_attn=True)
        else:
            x = self.encoder_blocks[-1](x)

        encoder_output = self.encoder_ln(x)

        B, T = decoder_input_ids.shape
        x = self.decoder_embd(decoder_input_ids)
        x = x + self.decoder_pos_embd(torch.arange(T, device=x.device))

        causal_mask = self.generate_causal_mask(T, device=x.device)

        for block in self.decoder_blocks[:-1]:
            x = block(x, encoder_output, causal_mask, padding_mask=pad_mask)

        if return_attn:
            x, decoder_attn = self.decoder_blocks[-1](
                x, encoder_output, causal_mask, padding_mask=pad_mask, return_attn=True
            )
        else:
            x = self.decoder_blocks[-1](
                x, encoder_output, causal_mask, padding_mask=pad_mask
            )
        x = self.decoder_ln(x)
        logits = self.head(x)
        if return_attn:
            return logits, {
                "encoder_self_attn": encoder_attn["self_attn"],
                "decoder_cross_attn": decoder_attn["cross_attn"],
            }
        return logits
