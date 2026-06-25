"""
Informer Encoder with ConvPool Distilling (Zhou et al., AAAI 2021)

The Distilling operation is the second key innovation of Informer.
After each ProbSparse attention layer, the sequence length is halved
using a 1D MaxPool. This creates a memory bottleneck that forces
the model to preserve only the most important temporal patterns.

Encoder stack: L layers, each halving the sequence:
  L_in → L_in/2 → L_in/4 → ... → L_in/2^(J-1)
"""

import torch
import torch.nn as nn
import math
from model.attention import ProbSparseSelfAttention


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Distilling Conv Layer
# ─────────────────────────────────────────────

class ConvDistilling(nn.Module):
    """
    Halves the sequence length after each attention layer.
    Conv1D (kernel=3) → ELU → MaxPool1D (stride=2)
    Input:  [B, L, d_model]
    Output: [B, L//2, d_model]
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.act  = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [B, L, d_model] → transpose for conv1d → [B, d_model, L]
        x = x.transpose(1, 2)
        x = self.act(self.norm(self.conv(x)))
        x = self.pool(x)
        return x.transpose(1, 2)  # [B, L//2, d_model]


# ─────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Single Encoder Layer
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor=5, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_w = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_w


# ─────────────────────────────────────────────
# Full Encoder Stack with Distilling
# ─────────────────────────────────────────────

class InformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff,
                 factor=5, dropout=0.1, distil=True):
        """
        Args:
            n_layers : number of encoder layers
            distil   : whether to use ConvDistilling between layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])

        self.distil_layers = nn.ModuleList([
            ConvDistilling(d_model)
            for _ in range(n_layers - 1)
        ]) if distil else None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        Returns:
            x        : [B, seq_len // 2^(n_layers-1), d_model]
            attn_list: list of attention weights per layer
        """
        attn_list = []
        for i, layer in enumerate(self.layers):
            x, attn_w = layer(x)
            attn_list.append(attn_w)
            if self.distil_layers and i < len(self.distil_layers):
                x = self.distil_layers[i](x)

        return self.norm(x), attn_list
