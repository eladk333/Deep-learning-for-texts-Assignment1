from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    return nn.Linear(input_vector_dim, 3* (input_vector_dim // n_heads)) 

def kqv(x, linear):    
    B, N, D = x.size()
    kqv_combined = linear(x) # size is (b, n, 3*d)
    size = kqv_combined.size(-1) // 3
    k, q, v = torch.split(kqv_combined, size, dim=-1) # size of each is (b, n, d)
    return k, q, v

def attention_scores(a, b):    
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    k_transposed = a.transpose(1, 2) # changes from (B, N1, D1) to (B, D1, N1)

    raw_scores = b @ k_transposed # size is (B, N2, N1)

    A = raw_scores / math.sqrt(D1) # Scaling
    
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

    sa = attention_weights @ v

    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask, out_matrix=None):
    
    B, N, D = x.size()

    head_outputs = []

    # Loop through each head's distinct linear layer
    for kqv_matrix in kqv_matrices:
        sa_head = self_attention_layer(x, kqv_matrix, mask)
        head_outputs.append(sa_head)

    sa = torch.cat(head_outputs, dim=-1)
    
    assert sa.size() == x.size()
    
    if out_matrix is not None:
        sa = out_matrix(sa)
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        return sa
