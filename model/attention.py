"""
ProbSparse Self-Attention — Core innovation of Informer (Zhou et al., AAAI 2021)

Standard Transformer self-attention: O(L²) complexity
ProbSparse attention: O(L log L) complexity

Key insight: In self-attention, only a few "dominant" queries
contribute meaningfully. The rest are near-uniform distributions
and add noise. ProbSparse selects the Top-u queries based on
a sparsity measurement M(qi, K) and computes attention only for those.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        """
        Args:
            d_model  : model dimension
            n_heads  : number of attention heads
            factor   : sampling factor c (controls how many queries are selected)
                       u = c * log(L_K) queries selected from L_Q total
            dropout  : attention dropout
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.factor   = factor

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Compute the sparsity measurement M(qi, K) for each query.

        Instead of computing full Q·K^T (which is O(L²)),
        we approximate by sampling sample_k keys for each query,
        computing a reduced attention score, and ranking queries
        by how 'dominant' their distribution is.

        M(qi, K) = max_j(qi·kj^T) - (1/L_K) * sum_j(qi·kj^T)
                 = measure of how far query i is from uniform distribution

        High M → query has a dominant key → important query
        Low M  → query is nearly uniform → can be replaced by average V
        """
        B, H, L_K, d_k = K.shape
        _, _, L_Q, _    = Q.shape

        # Sample sample_k keys randomly for each query
        K_sample_idx = torch.randint(L_K, (L_Q, sample_k))  # [L_Q, sample_k]
        K_sample = K[:, :, K_sample_idx, :]                  # [B, H, L_Q, sample_k, d_k]

        # Compute Q·K_sample^T for each query
        Q_exp = Q.unsqueeze(-2)                               # [B, H, L_Q, 1, d_k]
        QK = torch.matmul(Q_exp, K_sample.transpose(-2, -1)) # [B, H, L_Q, 1, sample_k]
        QK = QK.squeeze(-2)                                   # [B, H, L_Q, sample_k]

        # Sparsity measurement M
        M = QK.max(-1)[0] - QK.mean(-1)                      # [B, H, L_Q]

        # Select top-u queries
        M_top_idx = M.topk(n_top, dim=-1)[1]                 # [B, H, n_top]

        return M_top_idx

    def _get_initial_context(self, V, L_Q):
        """
        For non-selected (low-M) queries, use mean of V as default context.
        This is the key efficiency trick — instead of computing attention
        for all queries, we fill the rest with mean(V).
        """
        V_mean = V.mean(dim=-2)                               # [B, H, d_k]
        context = V_mean.unsqueeze(-2).expand(
            -1, -1, L_Q, -1
        ).clone()                                             # [B, H, L_Q, d_k]
        return context

    def _update_context(self, context, V, scores, top_idx):
        """
        Update context only for the selected top-u queries
        using full attention over all keys.
        """
        B, H, L_V, d_k = V.shape
        attn = torch.softmax(scores, dim=-1)                  # [B, H, n_top, L_V]
        attn = self.dropout(attn)

        # context_in: gather rows corresponding to top queries
        # Then scatter updated values back
        context_top = torch.matmul(attn, V)                   # [B, H, n_top, d_k]

        # Scatter back into context at top_idx positions
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, d_k)  # [B, H, n_top, d_k]
        context.scatter_(-2, top_idx_exp, context_top)

        return context

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Args:
            Q_in : [B, L_Q, d_model]
            K_in : [B, L_K, d_model]
            V_in : [B, L_V, d_model]
        Returns:
            output  : [B, L_Q, d_model]
            attn_w  : attention weights (for visualization)
        """
        B, L_Q, _ = Q_in.shape
        _, L_K, _ = K_in.shape

        # Project and reshape to [B, H, L, d_k]
        Q = self.W_Q(Q_in).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K_in).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V_in).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # Number of queries to select: u = factor * ln(L_K)
        sample_k = min(int(self.factor * math.log(L_K)), L_K)
        n_top    = min(int(self.factor * math.log(L_Q)), L_Q)

        # Step 1: Find dominant queries
        top_idx = self._prob_QK(Q, K, sample_k, n_top)       # [B, H, n_top]

        # Step 2: Gather top queries
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        Q_top = Q.gather(dim=2, index=top_idx_exp)            # [B, H, n_top, d_k]

        # Step 3: Compute full attention scores only for top queries
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q_top, K.transpose(-2, -1)) / scale  # [B, H, n_top, L_K]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Initialize context with mean(V), update top queries
        context = self._get_initial_context(V, L_Q)           # [B, H, L_Q, d_k]
        context = self._update_context(context, V, scores, top_idx)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        output  = self.W_O(context)

        # Return averaged attention weights for visualization
        attn_w = torch.softmax(scores, dim=-1).mean(dim=1)    # [B, n_top, L_K]

        return output, attn_w


class FullAttention(nn.Module):
    """Standard O(L²) attention — used in decoder cross-attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_in, K_in, V_in, mask=None):
        B, L_Q, _ = Q_in.shape
        _, L_K, _ = K_in.shape

        Q = self.W_Q(Q_in).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K_in).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V_in).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.W_O(out), attn.mean(dim=1)
