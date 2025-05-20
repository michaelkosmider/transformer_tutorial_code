import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_config)
        self.decoder = TransformerDecoder(**decoder_config)
        self.project = nn.Linear(
            decoder_config["hidden_size"], decoder_config["vocab_size"]
        )

    def forward(
        self, X_tgt, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X_src = self.encoder(X_src, src_key_padding_mask)

        X_tgt = self.decoder(
            X_tgt, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
        )

        return self.project(X_tgt)
