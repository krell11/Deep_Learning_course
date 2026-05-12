import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_rope_angles(dim, max_seq_len=100, theta=10000.0):
    """
    Precomputes the cosine and sine matrices for RoPE.
    """
    # Create position array: [0, 1, 2, ..., max_seq_len-1]
    position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
    
    # Create frequency divisors
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(theta) / dim))
    
    # Calculate angles: [max_seq_len, dim/2]
    angles = position * div_term
    
    # Duplicate angles for both halves of the embedding: [max_seq_len, dim]
    angles = torch.cat([angles, angles], dim=-1)
    
    # Return cos and sin
    return torch.cos(angles), torch.sin(angles)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to Queries and Keys."""
    # cos and sin shapes: [seq_len, head_dim]
    # q and k shapes: [batch, seq_len, num_heads, head_dim]
    
    # Reshape cos/sin to broadcast across batch and num_heads
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    
    return q_rotated, k_rotated


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, cos, sin, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # 2. Apply RoPE to Q and K!
        # Slice cos and sin to the current sequence length
        cos_seq = cos[:seq_len, :].to(x.device)
        sin_seq = sin[:seq_len, :].to(x.device)
        q, k = apply_rotary_pos_emb(q, k, cos_seq, sin_seq)
        
        # 3. Compute Attention Scores (transpose for batched matrix multiplication)
        # q, k shape: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # scores = (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply Causal Mask
        if mask is not None:
            # Expand mask to match num_heads
            scores = scores.masked_fill(mask, float('-inf'))
            
        # 5. Softmax and multiply by V
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # 6. Recombine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class RoPEDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256):
        super().__init__()
        self.attention = RoPEMultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, x, cos, sin, mask=None):
        # Attention with residual connection
        attn_out = self.attention(self.norm1(x), cos, sin, mask)
        x = x + attn_out
        
        # Feed Forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

