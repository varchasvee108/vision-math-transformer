import torch
import torch.nn as nn
from torch.nn import functional as F
from core.config import Config


class PatchEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.img_size = config.data.image_size
        self.n_embd = config.model.n_embd
        self.patch_size = config.model.patch_size
        self.grid_size = self.img_size[0] // self.patch_size
        self.row_embd = nn.Embedding(self.grid_size, self.n_embd)
        self.col_embd = nn.Embedding(self.grid_size, self.n_embd)
        self.ln = nn.LayerNorm(self.n_embd)
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_embd,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)

        gh, gw = self.grid_size, self.grid_size

        rows = self.row_embd(torch.arange(gh, device=x.device)).view(gh, 1, self.n_embd)
        cols = self.col_embd(torch.arange(gw, device=x.device)).view(1, gw, self.n_embd)

        pos_embd_2d = (rows + cols).view(-1, self.n_embd)

        return self.ln(x + pos_embd_2d)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_embd = config.model.n_embd
        out_dim = config.model.dim_ratio * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, out_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(out_dim, n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.model.n_embd
        self.n_head = config.model.num_heads
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.n_embd,
            num_heads=self.n_head,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x, return_attn=False, mask=None):
        x_norm = self.ln1(x)

        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, need_weights=return_attn, attn_mask=mask
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))

        if return_attn:
            return x, attn_weights
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        n_embd = config.model.n_embd
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            n_embd,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x, encoder_image, mask=None, padding_mask=None):
        x_norm = self.ln1(x)

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x_norm = self.ln2(x)
        attn_out, _ = self.cross_attn(x_norm, encoder_image, encoder_image)
        x = x + self.dropout(attn_out)
        x_norm = self.ln3(x)
        x = x + self.dropout(self.mlp(x_norm))
        return x
