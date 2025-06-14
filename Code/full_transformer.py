import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from functions import get_causal_mask


class Transformer(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.vocab_size = decoder_config["vocab_size"]  # Needed for generate.
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
        causal_mask = total_causal_mask[:1, :1]
        tgt_key_padding_mask = decoder_in == PAD_IDX
        features = self.decoder(
            decoder_in,
            src_features,
            causal_mask,
            tgt_key_padding_mask,
            src_key_padding_mask,
        )
        logits = self.project(features)

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
            causal_mask = total_causal_mask[:beam_cur_len, :beam_cur_len]
            tgt_key_padding_mask = decoder_in == PAD_IDX
            features = self.decoder(
                decoder_in,
                src_features,
                causal_mask,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )
            logits = self.project(features)
            next_token_logits = logits[:, beam_cur_len - 1, :]

            # Select the top K beams from the pool of candidates, update scores.
            candidates = beam_scores + next_token_logits
            scores, indices = torch.topk(candidates.view(-1), beam_K)
            beam_indices = indices // self.vocab_size
            tokens = indices % self.vocab_size

            beams = beams[
                beam_indices
            ]  # Re-arrange the beams in terms of score. Some beams are eliminated, while some are duplicated one or more times.
            beams[:, beam_cur_len] = tokens
            beam_scores = beam_scores[beam_indices] + scores.view(beam_K, 1)

        return beams[0]
