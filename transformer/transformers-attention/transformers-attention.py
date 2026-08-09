import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """

    
    
    transapose=torch.matmul(Q, K.transpose(-2, -1))
    d_k = K.shape[-1]
    
    sqrt=math.sqrt(d_k)
    
    logits=transapose/sqrt
    
    probs = torch.softmax(logits, dim=-1)

    output = torch.matmul(probs, V)

    return output
    # Your code here
    pass