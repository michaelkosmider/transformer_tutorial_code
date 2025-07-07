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

    def forward(self, X_Q, X_KV, causal_mask=None, key_padding_mask=None):
        #
        Q = slice_vertically(X_Q @ self.W_Q, self.key_size)
        K = slice_vertically(X_KV @ self.W_K, self.key_size)
        V = slice_vertically(X_KV @ self.W_V, self.value_size)

        A = compute_attention_matrix(Q, K, causal_mask, key_padding_mask)

        Y_prime = A @ V

        Y = unslice_vertically(Y_prime) @ self.W_O

        return Y
