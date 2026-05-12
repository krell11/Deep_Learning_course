import torch
from torch import nn
from .blocks.rope import RoPEDecoderBlock, precompute_rope_angles


class RoPEDecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=4, max_seq_len=100, pad_idx=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)

        # Precompute RoPE frequencies for the head dimension
        head_dim = d_model // nhead
        cos, sin = precompute_rope_angles(head_dim, max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
        self.layers = nn.ModuleList([
            RoPEDecoderBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x, dummy_tgt=None):
        seq_len = x.size(1)
        mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # 1. Token embeddings (Notice NO positional encoding is added here!)
        out = self.embedding(x)
        
        # 2. Pass through Decoder Blocks, providing the RoPE frequencies
        for layer in self.layers:
            out = layer(out, self.cos, self.sin, mask=mask)

        out = self.norm_final(out)
        return self.fc_out(out)

