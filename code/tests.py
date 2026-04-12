import torch
import attention

def test_attention_scores():
    B = 1  # Batch size
    N = 2  # Sequence length
    D = 4  # Embedding dimension

    a = torch.ones(B, N, D)
    b = torch.ones(B, N, D)
    
    expected_output = torch.ones(B, N, N) * 2.0    

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)


def test_self_attention():
    B = 1  # Batch size
    N = 2  # Sequence length
    D = 4  # Embedding dimension

    A = torch.zeros(B, N, N)
    v = torch.ones(B, N, D)
    
    expected_output = torch.ones(B, N, D)    

    actual_output = attention.self_attention(v, A)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(actual_output, expected_output)

if __name__ == "__main__":
    test_attention_scores()
    test_self_attention()
    print("All tests passed!")