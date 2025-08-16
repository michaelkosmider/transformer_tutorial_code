# transformer_components_michaelkosmider

This package contains the source code for the Transformer implementation featured in my tutorial:  
🔗 [Transformer Tutorial](https://michaelkosmider.github.io/transformertutorial/)

The full source code for this package is available here:  
[GitHub – transformer_components_michaelkosmider](https://github.com/michaelkosmider/transformer_tutorial_code/blob/main/README.md)

---

## Installation

This package requires PyTorch but does **not** install it automatically. You must install the correct version for your system (CPU/GPU, CUDA version, etc.) first:  
[PyTorch Installation Instructions](https://pytorch.org/get-started/locally/)

Once PyTorch is installed, install this package from PyPI.

## Available Imports

This package exposes the following classes and functions at the top level:

- `TransformerDecoder`
- `TransformerDecoderLayer`
- `TransformerEncoder`
- `TransformerEncoderLayer`
- `TransformerEncoderDecoder`
- `FeedForwardSubLayer`
- `AttentionSubLayer`
- `MultiHeadAttention`
- `compute_attention_matrix`
- `slice_vertically`
- `unslice_vertically`
- `get_causal_mask`

Example:
```python
from transformer_components_michaelkosmider import TransformerEncoder, get_causal_mask

encoder = TransformerEncoder(...)
mask = get_causal_mask(seq_len=10)