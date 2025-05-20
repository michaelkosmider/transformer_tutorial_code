import torch


def compute_attention_matrix(Q, K, causal_mask=None, key_padding_mask=None):
    """
    Private notes: expecting Q and K to be of shapes (N_batch, heads, Qseqlen, D_k) and (N_batch, heads, Kseqlen, D_k)
    """
    E = Q @ K.transpose(-1, -2)

    if causal_mask is not None:
        E.masked_fill_(causal_mask, -torch.inf)

    if key_padding_mask is not None:
        E.masked_fill_(key_padding_mask[:, torch.newaxis, torch.newaxis, :], -torch.inf)

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

Suppose the input is a 3 dimensional tensor of shape (S, R, C). This function will treat the 
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
