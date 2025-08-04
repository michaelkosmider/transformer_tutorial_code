import torch
import torch.nn as nn


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, custom_encoder, custom_decoder):
        super().__init__()

        self.encoder = custom_encoder
        self.decoder = custom_decoder

    def forward(
        self, X_tgt, X_src, tgt_mask, tgt_key_padding_mask, src_key_padding_mask
    ):

        X_src = self.encoder(X_src, src_key_padding_mask)

        logits = self.decoder(
            X_tgt, X_src, tgt_mask, tgt_key_padding_mask, src_key_padding_mask
        )

        return logits

    # Quick note: the decoder must expose .stack_size, .vocab_size, .key_size and .value_size, .num_heads
    def generate(
        self,
        X_src,
        src_key_padding_mask,
        num_beams,
        max_beam_len,
        SOS_IDX,
        PAD_IDX,
        EOS_IDX,
    ):

        batch_size = X_src.shape[0]

        # Initialize the kv cache
        all_kv_cache = self.initialize_kv_cache(
            num_beams, max_beam_len, device=X_src.device
        )

        # Get source features (encoder output). Only need to do this once.
        src_features = self.encoder(X_src, src_key_padding_mask)
        # Repeat source once per beam. Shape goes from (N,T,H) to (N*K,T,H).
        src_features = torch.repeat_interleave(src_features, num_beams, dim=0)

        # Initialize empty beams
        beams = torch.full((num_beams, max_beam_len), PAD_IDX, device=X_src.device)
        beams[:, 0] = SOS_IDX

        # Get logits, tokens and scores for the SOS token distribution.
        logits = self.decoder(
            beams[:, 0:1],
            src_features,
            src_key_padding_mask=src_key_padding_mask,
            all_kv_cache=all_kv_cache,
            position=0,
        )
        vocab_scores = torch.log_softmax(logits[0], dim=-1).view(-1)
        beam_scores, tokens = torch.topk(vocab_scores, num_beams)
        beams[:, 1] = tokens

        # For keeping track of finished beams.
        dead_beams = tokens == EOS_IDX

        for position in range(1, max_beam_len - 1):
            logits = self.decoder(
                beams[:, position : position + 1],
                src_features,
                src_key_padding_mask=src_key_padding_mask,
                all_kv_cache=all_kv_cache,
                position=position,
            ).squeeze()

            log_probs = torch.log_softmax(logits, -1)
            log_probs[dead_beams] = -torch.inf
            log_probs[dead_beams, EOS_IDX] = 0
            candidate_scores = beam_scores.view(num_beams, 1) + log_probs

            beam_scores, indices = torch.topk(candidate_scores.view(-1), num_beams)
            next_tokens = indices % self.decoder.vocab_size
            next_beam_indices = indices // self.decoder.vocab_size

            beams = beams[next_beam_indices]
            beams[:, position + 1] = next_tokens

            dead_beams = dead_beams[next_beam_indices] | (next_tokens == EOS_IDX)
            if dead_beams.all():
                break

            self.reorder_kv_cache(all_kv_cache, next_beam_indices)

        return beams

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
