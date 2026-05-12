import torch
from torch import nn
from .blocks.pos_enc import PositionalEncoding


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=4, pad_idx=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x, dummy_tgt=None): 
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        padding_mask = (x == self.pad_idx)
        
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)

        out = self.transformer_decoder(
            x_emb, mask=mask, src_key_padding_mask=padding_mask
        )
        return self.fc_out(out)
