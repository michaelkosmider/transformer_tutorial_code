import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention


class AttentionSubLayer(nn.Module):
    def __init__(self, num_heads, hidden_size, key_size, value_size, dropout):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, hidden_size, key_size, value_size
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    # For self attention, X_Q and X_KV are the same. For cross attention, X_KV comes from the reference sequence.
    def forward(self, X_Q, X_KV, tgt_mask=None, key_padding_mask=None, kv_cache=None):
        return self.layernorm(
            self.dropout(
                self.multihead_attention(
                    X_Q,
                    X_KV,
                    tgt_mask=tgt_mask,
                    key_padding_mask=key_padding_mask,
                    kv_cache=kv_cache,
                )
            )
            + X_Q
        )


class FeedForwardSubLayer(nn.Module):
    """The feedforward network is a two layer mlp with a relu activation in between.
        Dropout applied to MLP output and residual connected added. Layer norm applied
        before output.

    Args:
        hidden_size (int): Size of the hidden dimension.
        feedforward_size (int): Size of the hidden layer of the MLP.
        dropout (float): Dropout rate, which is applied after the MLP.
    """

    def __init__(self, hidden_size, feedforward_size, dropout):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, X):
        return self.layernorm(self.dropout(self.feedforward(X)) + X)
