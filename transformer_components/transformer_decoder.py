import torch
import torch.nn as nn
from .sublayers import AttentionSubLayer, FeedForwardSubLayer


class TransformerDecoderLayer(nn.Module):

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
        self,
        X_Q,
        X_KV,
        tgt_mask,
        tgt_key_padding_mask,
        src_key_padding_mask,
        layer_kv_cache,
    ):

        X = self.self_multihead_attention_sublayer(
            X_Q,
            X_Q,
            tgt_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            kv_cache=layer_kv_cache["tgt"] if layer_kv_cache is not None else None,
        )

        X = self.cross_multihead_attention_sublayer(
            X,
            X_KV,
            key_padding_mask=src_key_padding_mask,
            kv_cache=layer_kv_cache["src"] if layer_kv_cache is not None else None,
        )

        X = self.feedforward_sublayer(X)

        return X


class TransformerDecoder(nn.Module):

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

        self.decoder_stack = nn.ModuleList(
            [
                TransformerDecoderLayer(
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
        self,
        X_tgt,
        X_src,
        tgt_mask,
        tgt_key_padding_mask,
        src_key_padding_mask,
        all_kv_cache=None,
    ):

        for (
            i,
            decoder_layer,
        ) in enumerate(self.decoder_stack):

            layer_kv_cache = all_kv_cache[i] if all_kv_cache is not None else None

            X_tgt = decoder_layer(
                X_tgt,
                X_src,
                tgt_mask,
                tgt_key_padding_mask,
                src_key_padding_mask,
                layer_kv_cache,
            )

        return X_tgt
