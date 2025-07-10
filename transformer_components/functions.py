import torch

"""
Desciption:

This function computes the attention matrix between queries Q and keys K, which have shapes 
(T, D_K) and (S, D_K) respectively. Here, T is the length of the input sequence used to create
Q, and S is the length of the source sequence producing K and V. Forthemore, D_K is the key 
size. The attention computation is performed in batches of size N_batch, heads, which correspond 
to the number of sequences in the batch, and the number of heads for each sequence.

Input:

Q - a tensor of shape (N_batch, heads, T, D_K)
K - a tensor of shape (N_batch, heads, S, D_K)
causal_mask - a T by T boolean causal mask, where True means do not attend. 
key_padding_mask - an N by S boolean mask, where True means do not attend. 

Output:

A - a tensor of shape (N_batch, heads, T, S)
"""


def compute_attention_matrix(Q, K, tgt_mask=None, key_padding_mask=None):

    E = Q @ K.transpose(-1, -2)

    if tgt_mask is not None:
        E.masked_fill_(tgt_mask, -torch.inf)

    if key_padding_mask is not None:
        E.masked_fill_(key_padding_mask[:, None, None, :], -torch.inf)

    A = torch.softmax(E / (Q.shape[-1] ** 0.5), -1)
    return A


"""
Desciption:

This function creates vertical slices out of a 2 dimensional tensor. For a tensor with R rows 
and C columns, using a slice size of S yields a tensor of shape (C/S, R, S), where 
the first dimension specifies the slice. Furthermore, any batch size (d0, d1, ... , dn) is 
supported. The slice size (number of columns in the slice) must divide the number of columns. 

Input:

X - a tensor of shape (d0, d1, ... , dn, R, C)

Output:

- a tensor of shape (d0, d1, ... , dn, C/S, R, slice_size)
"""


def slice_vertically(X, slice_size):
    return X.unflatten(dim=-1, sizes=(-1, slice_size)).transpose(-2, -3)


"""
Desciption:

The input is a 3 dimensional tensor of shape (S, R, C). This function will treat the 
input as a list of S tensors of shape (R, C) and concatenate them along the column dimension,
resulsting in a tensor of shape (R, C x S). It undoes the result from slice_vertically, meaning 
that X = unslice_vertically(slice_vertically(X). Furthermore, any batch size (d0, d1, ... , dn) 
is supported.

Input:

X - a tensor of shape (d0, d1, ... , dn, S, R, C)

Output:

- a tensor of shape (d0, d1, ... , dn, R, C x S)
"""


def unslice_vertically(X):
    return X.transpose(-2, -3).flatten(-2, -1)


"""
Desciption:

Returns a T by T causal mask on the specified device.
"""


def get_causal_mask(T, device):
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
