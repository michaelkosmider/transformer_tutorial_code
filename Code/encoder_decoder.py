import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from functions import get_causal_mask


# Wrapper for the TransformerEncoder that embeds tokens and adds positional encodings.
class Encoder(nn.Module):

    def __init__(self, transformer_encoder_config, vocab_size, context_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, transformer_encoder_config["hidden_size"]
        )
        self.positional_encoding = nn.Embedding(
            context_size, transformer_encoder_config["hidden_size"]
        )
        self.transformer_encoder = TransformerEncoder(**transformer_encoder_config)

    def forward(self, X, key_padding_mask):

        X = self.embedding(X) + self.positional_encoding(
            torch.arange(X.shape[1], device=X.device)
        )
        X = self.transformer_encoder(X, key_padding_mask)

        return X


# Similar idea to above.
class Decoder(nn.Module):

    def __init__(self, transformer_decoder_config, vocab_size, context_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, transformer_decoder_config["hidden_size"]
        )
        self.positional_encoding = nn.Embedding(
            context_size, transformer_decoder_config["hidden_size"]
        )
        self.project = nn.Linear(
            transformer_decoder_config["hidden_size"],
            vocab_size,
        )
        self.transformer_decoder = TransformerDecoder(**transformer_decoder_config)

    def forward(
        self, X_tgt, X_src, tgt_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X_tgt = self.embedding(X_tgt) + self.positional_encoding(
            torch.arange(X_tgt.shape[1], device=X_tgt.device)
        )
        features = self.transformer_decoder(
            X_tgt, X_src, tgt_mask, tgt_key_padding_mask, src_key_padding_mask
        )
        logits = self.project(features)

        return logits


class EncoderDecoder(nn.Module):

    def __init__(self, custom_encoder, custom_decoder):
        super().__init__()

        self.encoder = custom_encoder
        self.decoder = custom_decoder

    def forward(
        self, X_tgt, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X_src = self.encoder(X_src, src_key_padding_mask)

        logits = self.decoder(
            X_tgt, X_src, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask
        )

        return logits

    def generate(
        self, X_src, src_key_padding_mask, beam_K, beam_max_len, SOS_IDX, PAD_IDX
    ):

        device = X_src.device

        # Get source features (encoder output). Only need to do this once.
        src_features = self.encoder(X_src, src_key_padding_mask)

        # Initialize empty beams with SOS + PAD
        beams = torch.full(
            (beam_K, beam_max_len), PAD_IDX, dtype=torch.int, device=device
        )
        beams[:, 0] = SOS_IDX
        total_causal_mask = get_causal_mask(beam_max_len, device=device)

        # Obtain logits for a single SOS token, which will attend to the source sentence.
        decoder_in = beams[:1, :1]
        tgt_mask = total_causal_mask[:1, :1]
        tgt_key_padding_mask = decoder_in == PAD_IDX
        logits = self.decoder(
            decoder_in,
            src_features,
            tgt_mask,
            tgt_key_padding_mask,
            src_key_padding_mask,
        )

        # Obtain the score for each token. Also obtain a probability distribution over all tokens.
        first_token_probs = torch.softmax(logits, dim=-1).view(-1)
        first_token_scores = torch.log_softmax(logits, dim=-1).view(
            -1
        )  # Despite already having softmax, log_softmax is more numerically stable, and thus preferred.

        # Sample K tokens, and place them in the beams. Initialize scores for each beam.
        tokens = torch.multinomial(first_token_probs, beam_K)
        beams[:, 1] = tokens
        beam_scores = first_token_scores[tokens].view(beam_K, 1)

        # Deterministic beam search for remaining tokens.
        for beam_cur_len in range(2, beam_max_len):

            # Get logits for the next token (across all beams).
            decoder_in = beams[:, :beam_cur_len]
            tgt_mask = total_causal_mask[:beam_cur_len, :beam_cur_len]
            tgt_key_padding_mask = decoder_in == PAD_IDX
            logits = self.decoder(
                decoder_in,
                src_features,
                tgt_mask,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )
            next_token_logits = logits[:, beam_cur_len - 1, :]

            # Select the top K beams from the pool of candidates, update scores.
            candidates = beam_scores + next_token_logits
            scores, indices = torch.topk(candidates.view(-1), beam_K)
            beam_indices = indices // self.decoder.vocab_size
            tokens = indices % self.decoder.vocab_size

            beams = beams[
                beam_indices
            ]  # Re-arrange the beams in terms of score. Some beams are eliminated, while some are duplicated one or more times.
            beams[:, beam_cur_len] = tokens
            beam_scores = beam_scores[beam_indices] + scores.view(beam_K, 1)

        return beams[0]
