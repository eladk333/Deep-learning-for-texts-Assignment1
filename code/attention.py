from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3*head_dim) 
    

def kqv(x, linear):
    projected = linear(x) # size is (b x n x 3*head_dim)
    k, q, v = projected.chunk(3, dim=-1) # each of size (b x n x head_dim)
    return k, q, v

def attention_scores(a, b):
    

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    a_transposed = a.transpose(1, 2) # now size is (b x d x n)
    A = torch.bmm(b, a_transposed) / math.sqrt(D1) # size is (b x n x n)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    ones = torch.ones(max_context_len, max_context_len)
    mask = torch.tril(ones) # size is (max_context_len x max_context_len)
    # Add a dimension so the size is (1 x max_context_len x max_context_len)
    mask = mask.view(1, max_context_len, max_context_len)
    return mask

def self_attention(v, A, mask = None):

    if mask is not None:
        N = A.size(-1) 
        M = mask[:, :N, :N]

        A = A.masked_fill(M == 0, float("-inf"))
    
    attention_weights = F.softmax(A, dim=-1)
    sa = torch.bmm(attention_weights, v) # size is (b x n x head_dim)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    head_outputs = []

    for kqv_matrix in kqv_matrices:
        sa = self_attention_layer(x, kqv_matrix, mask)
        head_outputs.append(sa)

    sa = torch.cat(head_outputs, dim=-1) # concatenate the outputs of the heads along the last dimension 
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

        # Add the final linear projection layer (d dimensions to d dimensions)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        # Apply the final projection to the concatenated multi-head output
        output = self.out_proj(sa)
        return output
