import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

# sublayers


class AttentionSubLayer(nn.Module):
    def __init__(self, num_heads, hidden_size, key_size, value_size, dropout):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, hidden_size, key_size, value_size
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    # For self attention, X_Q and X_KV are the same. For cross attention, X_KV comes from the reference sequence.
    def forward(self, X_Q, X_KV):
        return self.layernorm(self.dropout(self.multihead_attention(X_Q, X_KV)) + X_Q)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout):
        super().__init__()
        # The feedforward network is a two layer mlp with a relu activation in between.
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, X):
        return self.layernorm(self.dropout(self.feedforward(X)) + X)


class EncoderLayer(nn.Module):
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

    def forward(self, X):
        return self.feedforward_sublayer(self.multihead_attention_sublayer(X, X))


# full encoder


class TransformerEncoder(nn.Module):
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
        self.encoder_stack = nn.ModuleList(
            [
                EncoderLayer(
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

    """    
    Description: Applies the transformer encoder to the input.
    Below, N is batch size, T is sequence length and H must be self.hidden_dim.

    inputs: 
        X - a tensor of shape N x T 
        
    outputs:
        returns a tensor of shape N x T x vocab_size
    """

    def forward(self, X):
        X = self.embeddings(X) + self.positional_encodings(
            torch.arange(X.shape[1], device=X.device)
        )

        for encoder_layer in self.encoder_stack:
            X = encoder_layer(X)

        return X
