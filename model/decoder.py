"""
Informer Decoder — Generative Style (Zhou et al., AAAI 2021)

Third key innovation: instead of step-by-step autoregressive decoding
(which is slow), the Informer decoder generates the entire forecast
in ONE forward pass.

Decoder input = [known label segment | zero padding for forecast]
The model fills in the zeros in one shot.

Each decoder layer has:
1. ProbSparse Self-Attention on decoder input
2. Full Cross-Attention between decoder and encoder output
3. Feed-Forward Network
"""

import torch
import torch.nn as nn
from model.attention import ProbSparseSelfAttention, FullAttention
from model.encoder import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor=5, dropout=0.1):
        super().__init__()
        # Self-attention on decoder sequence (masked)
        self.self_attn  = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        # Cross-attention between decoder queries and encoder keys/values
        self.cross_attn = FullAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        """
        x       : decoder input [B, label_len + pred_len, d_model]
        enc_out : encoder output [B, distilled_len, d_model]
        """
        # Masked self-attention (decoder attends to its own past)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention (decoder attends to encoder output)
        cross_out, cross_w = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + cross_out)

        x = self.norm3(x + self.ff(x))
        return x, cross_w


class InformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff,
                 pred_len, n_features, factor=5, dropout=0.1):
        """
        Args:
            pred_len   : forecast horizon (how many steps to predict)
            n_features : input feature dimension
        """
        super().__init__()
        self.pred_len = pred_len

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        # Project from d_model to scalar NEE prediction
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x, enc_out):
        """
        x       : decoder input [B, label_len + pred_len, d_model]
        enc_out : encoder memory [B, distilled_len, d_model]
        Returns:
            predictions : [B, pred_len, 1]
            cross_w     : cross-attention weights from last layer
        """
        cross_w = None
        for layer in self.layers:
            x, cross_w = layer(x, enc_out)

        x = self.norm(x)
        # Only take the forecast portion (last pred_len steps)
        x = x[:, -self.pred_len:, :]
        return self.projection(x), cross_w
