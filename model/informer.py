"""
Informer — Full Model Assembly (Zhou et al., AAAI 2021)
Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
AAAI 2021 Best Paper Award

Architecture summary:
  Input Embedding → Positional Encoding
       ↓
  Encoder (ProbSparse Attention + ConvDistilling × n_enc_layers)
       ↓
  Decoder (ProbSparse Self-Attn + Full Cross-Attn × n_dec_layers)
       ↓
  Linear Projection → NEE forecast
"""

import torch
import torch.nn as nn
from model.encoder import InformerEncoder, PositionalEncoding
from model.decoder import InformerDecoder


class Informer(nn.Module):
    def __init__(
        self,
        n_features,      # number of input features
        d_model=128,     # model dimension
        n_heads=8,       # attention heads
        n_enc_layers=3,  # encoder depth
        n_dec_layers=2,  # decoder depth
        d_ff=256,        # feed-forward dimension
        factor=5,        # ProbSparse sampling factor
        seq_len=96,      # encoder input length
        label_len=48,    # known decoder input length
        pred_len=24,     # forecast horizon
        dropout=0.1,
        distil=True      # use ConvDistilling in encoder
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        self.d_model   = d_model

        # ── Input Projections ──────────────────────────────────────
        # Project raw features to d_model
        self.enc_embedding = nn.Linear(n_features, d_model)
        self.dec_embedding = nn.Linear(n_features, d_model)

        # ── Positional Encoding ────────────────────────────────────
        self.enc_pos = PositionalEncoding(d_model, dropout=dropout)
        self.dec_pos = PositionalEncoding(d_model, dropout=dropout)

        # ── Encoder ───────────────────────────────────────────────
        self.encoder = InformerEncoder(
            n_layers=n_enc_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            distil=distil
        )

        # ── Decoder ───────────────────────────────────────────────
        self.decoder = InformerDecoder(
            n_layers=n_dec_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            pred_len=pred_len,
            n_features=n_features,
            factor=factor,
            dropout=dropout
        )

    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc : [B, seq_len, n_features]         encoder input
            x_dec : [B, label_len + pred_len, n_features] decoder input
                    (label portion = known, forecast portion = zeros)
        Returns:
            pred  : [B, pred_len, 1]  NEE forecast
        """
        # Encode
        enc_emb = self.enc_pos(self.enc_embedding(x_enc))
        enc_out, enc_attns = self.encoder(enc_emb)

        # Decode
        dec_emb = self.dec_pos(self.dec_embedding(x_dec))
        pred, cross_attn = self.decoder(dec_emb, enc_out)

        return pred, enc_attns, cross_attn

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(n_features, seq_len=96, label_len=48, pred_len=24, device="cpu"):
    """Factory function to build and initialize Informer."""
    model = Informer(
        n_features=n_features,
        d_model=64,         # smaller for CPU
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=1,
        d_ff=128,
        factor=5,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        dropout=0.1,
        distil=True
    ).to(device)

    print(f"Informer model built: {model.count_parameters():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Quick sanity check
    B, seq, label, pred, feat = 4, 96, 48, 24, 16
    model = build_model(n_features=feat, seq_len=seq, label_len=label, pred_len=pred)

    x_enc = torch.randn(B, seq, feat)
    x_dec = torch.randn(B, label + pred, feat)

    out, enc_attns, cross_attn = model(x_enc, x_dec)
    print(f"Output shape: {out.shape}")   # [4, 24, 1]
    print("Sanity check passed!")
