import torch
import torch.nn as nn
from sublayers import AttentionSubLayer, FeedForwardSubLayer


class DecoderLayer(nn.Module):

    def __init__(
        self, num_heads, hidden_size, key_size, value_size, feedforward_size, dropout
    ):
        super().__init__()

        self.self_multihead_attention_sublayer = AttentionSubLayer(
            num_heads, hidden_size, key_size, value_size, dropout
        )

        self.cross_multihead_attention_sublayer = AttentionSubLayer(
            num_heads, hidden_size, key_size, value_size, dropout
        )

        self.feedforward_sublayer = FeedForwardSubLayer(
            hidden_size, feedforward_size, dropout
        )

    def forward(
        self, X_Q, X_KV, causal_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X = self.self_multihead_attention_sublayer(
            X_Q, X_Q, causal_mask=causal_mask, key_padding_mask=tgt_key_padding_mask
        )

        X = self.cross_multihead_attention_sublayer(
            X, X_KV, key_padding_mask=src_key_padding_mask
        )

        X = self.feedforward_sublayer(X)

        return X


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_size,
        stack_size=6,
        num_heads=8,
        hidden_size=512,
        key_size=64,
        value_size=64,
        feedforward_size=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.positional_encodings = nn.Embedding(context_size, hidden_size)
        self.decoder_stack = nn.ModuleList(
            [
                DecoderLayer(
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

    def forward(
        self, X_tgt, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X = self.embeddings(X_tgt) + self.positional_encodings(
            torch.arange(X_tgt.shape[1], device=X_tgt.device)
        )

        for decoder_layer in self.decoder_stack:
            X = decoder_layer(
                X, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
            )

        return X
