import torch
from torch import nn


class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_dim=128, pad_idx=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.encoder = nn.LSTM(d_model, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(d_model, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        src_lengths = (src != self.pad_idx).sum(dim=1).cpu() 
        
        src_emb = self.embedding(src)
        
        # This creates a special object that tells the LSTM to ignore padding
        packed_src = nn.utils.rnn.pack_padded_sequence(
            src_emb, src_lengths, batch_first=True, enforce_sorted=False
        )
        
        # 3. Pass through encoder
        _, (hidden, cell) = self.encoder(packed_src)
        
        tgt_emb = self.embedding(tgt)
        out, _ = self.decoder(tgt_emb, (hidden, cell))
        return self.fc_out(out)

