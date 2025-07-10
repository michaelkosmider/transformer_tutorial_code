import torch
import torch.nn as nn
from .functions import get_causal_mask


class TransformerEncoderDecoder(nn.Module):

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
        self, X_src, src_key_padding_mask, num_beams, max_beam_len, SOS_IDX, PAD_IDX
    ):

        # Initialize the kv cache
        all_kv_cache = self.initialize_kv_cache(
            num_beams, max_beam_len, device=X_src.device
        )

        # Get source features (encoder output). Only need to do this once.
        src_features = self.encoder(X_src, src_key_padding_mask)

        # Initialize empty beams
        beams = torch.full((num_beams, max_beam_len), PAD_IDX, device=X_src.device)
        beams[:, 0] = SOS_IDX

        # Get logits, tokens and scores for the SOS token distribution.
        logits = self.decoder(
            beams[:, 0:1],
            src_features,
            src_key_padding_mask=src_key_padding_mask,
            all_kv_cache=all_kv_cache,
        )
        vocab_scores = torch.log_softmax(logits[0], dim=-1).view(-1)
        beam_scores, tokens = torch.topk(vocab_scores, num_beams)
        beam_scores = beam_scores.view(num_beams, 1)
        beams[:, 1] = tokens

        ###
        logits = self.decoder(
            beams[:, 1:2],
            src_features,
            src_key_padding_mask=src_key_padding_mask,
            all_kv_cache=all_kv_cache,
        ).squeeze()

        candidate_scores = beam_scores + torch.log_softmax(logits, -1)
        beam_scores, indices = torch.topk(candidate_scores.view(-1), num_beams)
        beam_scores = beam_scores.view(num_beams, 1)

        next_tokens = indices % self.decoder.vocab_size
        next_beam_indices = indices // self.decoder.vocab_size

        beams = beams[next_beam_indices]
        beams[:, 2] = next_tokens

        self.reorder_kv_cache(all_kv_cache, next_beam_indices)

        return all_kv_cache

    def initialize_kv_cache(self, num_beams, max_beam_len, device):

        all_kv_cache = [{} for i in range(self.decoder.stack_size)]

        for layer_kv_cache in all_kv_cache:
            # Cache for self attention
            tgt_kv_cache = {}
            tgt_kv_cache["mode"] = "self_attn"
            tgt_kv_cache["K"] = torch.zeros(
                size=(
                    num_beams,
                    max_beam_len,
                    self.decoder.num_heads * self.decoder.key_size,
                ),
                device=device,
            )
            tgt_kv_cache["V"] = torch.zeros(
                size=(
                    num_beams,
                    max_beam_len,
                    self.decoder.num_heads * self.decoder.value_size,
                ),
                device=device,
            )
            tgt_kv_cache["cache_len"] = 0

            # Cache for cross attention
            src_kv_cache = {}
            src_kv_cache["mode"] = "cross_attn"

            layer_kv_cache["tgt"] = tgt_kv_cache
            layer_kv_cache["src"] = src_kv_cache

        return all_kv_cache

    def reorder_kv_cache(self, all_kv_cache, beam_indices):

        for layer_kv_cache in all_kv_cache:
            layer_kv_cache["tgt"]["K"] = layer_kv_cache["tgt"]["K"][beam_indices]
            layer_kv_cache["tgt"]["V"] = layer_kv_cache["tgt"]["V"][beam_indices]

    # def generate(
    #     self, X_src, src_key_padding_mask, num_beams, max_beam_len, SOS_IDX, PAD_IDX
    # ):

    #     # Initialize the kv cache
    #     all_kv_cache = self.initialize_kv_cache(
    #         num_beams, max_beam_len, device=X_src.device
    #     )

    #     # Get source features (encoder output). Only need to do this once.
    #     src_features = self.encoder(X_src, src_key_padding_mask)

    #     # Initialize empty beams with SOS + PAD
    #     beams = torch.full(
    #         (num_beams, max_beam_len), PAD_IDX, dtype=torch.int, device=X_src.device
    #     )
    #     beams[:, 0] = SOS_IDX

    #     # Obtain logits for the first token. They are all the SOS token.
    #     decoder_in = beams[:, 0:1]  # shape : (num_beams, 1)
    #     logits = self.decoder(
    #         decoder_in,
    #         src_features,
    #         tgt_mask=None,
    #         tgt_key_padding_mask=None,
    #         src_key_padding_mask=src_key_padding_mask,
    #         all_kv_cache=all_kv_cache,
    #     )  # shape : (num_beams, 1, vocab_size)

    #     first_token_scores = torch.log_softmax(logits[0], dim=-1).view(-1)

    #     # Obtain the top num_beams tokens and place them in the beams
    #     scores, tokens = torch.topk(first_token_scores, num_beams)
    #     beams[:, 1] = tokens
    #     beam_scores = scores.view(num_beams, 1)

    #     # Beam search for remaining tokens.
    #     for beam_cur_len in range(2, max_beam_len):

    #         # Get logits for the next token (across all beams).
    #         decoder_in = beams[:, beam_cur_len - 1 : beam_cur_len]
    #         next_token_logits = self.decoder(
    #             decoder_in,
    #             src_features,
    #             tgt_mask=None,
    #             tgt_key_padding_mask=None,
    #             src_key_padding_mask=src_key_padding_mask,
    #             all_kv_cache=all_kv_cache,
    #         )[:, 0, :]

    #         # Select the top K beams from the pool of candidates, update scores.
    #         candidate_scores = beam_scores + torch.log_softmax(
    #             next_token_logits, dim=-1
    #         )
    #         scores, indices = torch.topk(candidate_scores.view(-1), num_beams)
    #         beam_indices = indices // self.decoder.vocab_size
    #         tokens = indices % self.decoder.vocab_size

    #         # Re-arrange the beams in terms of score. Some beams may be eliminated, in which case some others are duplicated one or more times.
    #         beams = beams[beam_indices]
    #         beams[:, beam_cur_len] = tokens
    #         beam_scores = scores.view(num_beams, 1)

    #         # Re-arrange caches based on which beams were selected.
    #         self.reorder_kv_cache(all_kv_cache, beam_indices)

    #     return beams
