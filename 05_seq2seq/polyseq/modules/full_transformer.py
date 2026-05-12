import torch
from torch import nn
from .blocks.pos_enc import PositionalEncoding


class FullTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, pad_idx=None):
        super().__init__()
        assert self.pad_idx is not None
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(out)

