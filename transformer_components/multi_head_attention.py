import torch
import torch.nn as nn
from .functions import compute_attention_matrix, slice_vertically, unslice_vertically


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, key_size, value_size):
        super().__init__()

        self.key_size = key_size
        self.value_size = value_size

        self.W_Q = nn.Parameter(torch.empty(hidden_size, key_size * num_heads))
        self.W_K = nn.Parameter(torch.empty(hidden_size, key_size * num_heads))
        self.W_V = nn.Parameter(torch.empty(hidden_size, value_size * num_heads))
        self.W_O = nn.Parameter(torch.empty(value_size * num_heads, hidden_size))

        for param in self.parameters():
            nn.init.xavier_normal_(param)

    def forward(
        self,
        X_Q,
        X_KV,
        tgt_mask=None,
        key_padding_mask=None,
        kv_cache=None,
    ):

        if kv_cache is None:
            Q = X_Q @ self.W_Q
            K = X_KV @ self.W_K
            V = X_KV @ self.W_V

        elif kv_cache["mode"] == "cross_attn":
            Q = X_Q @ self.W_Q

            if (
                "K" not in kv_cache
            ):  # Could also check for V, doesn't matter. Happens once, the first time the src KVs are computed.
                kv_cache["K"] = X_KV @ self.W_K
                kv_cache["V"] = X_KV @ self.W_V

            K = kv_cache["K"]
            V = kv_cache["V"]

        elif kv_cache["mode"] == "self_attn":

            # Compute new Query, Key, and Value (for just one token).
            Q = X_Q @ self.W_Q
            K_new = X_KV @ self.W_K
            V_new = X_KV @ self.W_V

            # Retrieve the cache.
            K = kv_cache["K"]
            V = kv_cache["V"]

            # Update the cache with new token Key and Value.
            K[:, kv_cache["cache_len"], :] = K_new[:, 0, :]
            V[:, kv_cache["cache_len"], :] = V_new[:, 0, :]
            kv_cache["cache_len"] += 1

            # Select the valid portion of the cache.
            K = K[:, : kv_cache["cache_len"], :]
            V = V[:, : kv_cache["cache_len"], :]

        Q = slice_vertically(Q, self.key_size)
        K = slice_vertically(K, self.key_size)
        V = slice_vertically(V, self.value_size)

        A = compute_attention_matrix(Q, K, tgt_mask, key_padding_mask)

        Y_prime = A @ V

        Y = unslice_vertically(Y_prime) @ self.W_O

        return Y
