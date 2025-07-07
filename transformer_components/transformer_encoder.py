import torch
import torch.nn as nn
from .sublayers import AttentionSubLayer, FeedForwardSubLayer


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self, num_heads, hidden_size, key_size, value_size, feedforward_size, dropout
    ):
        super().__init__()

        self.multihead_attention_sublayer = AttentionSubLayer(
            num_heads, hidden_size, key_size, value_size, dropout
        )

        self.feedforward_sublayer = FeedForwardSubLayer(
            hidden_size, feedforward_size, dropout
        )

    def forward(self, X, key_padding_mask):

        X = self.multihead_attention_sublayer(X, X, key_padding_mask=key_padding_mask)

        X = self.feedforward_sublayer(X)

        return X


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        stack_size=6,
        num_heads=8,
        hidden_size=512,
        key_size=64,
        value_size=64,
        feedforward_size=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder_stack = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_heads,
                    hidden_size,
                    key_size,
                    value_size,
                    feedforward_size,
                    dropout,
                )
                for _ in range(stack_size)
            ]
        )

    def forward(self, X, key_padding_mask):

        for encoder_layer in self.encoder_stack:
            X = encoder_layer(X, key_padding_mask)

        return X
