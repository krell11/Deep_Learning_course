from functools import singledispatch
from .dec_transformer import DecoderOnlyTransformer
from .full_transformer import FullTransformer
from .rope_dec_transformer import RoPEDecoderOnlyTransformer
from .lstm import LSTMSeq2Seq


def is_encoder_decoder(model):
    match model:
        case DecoderOnlyTransformer() | RoPEDecoderOnlyTransformer():
            return False
        case FullTransformer() | LSTMSeq2Seq():
            return False
        case _:
            raise ValueError(f'Unknown type of model: {type(model)!r}')


@singledispatch
def get_next_logits(model, enc_inputs, dec_input, lstm_hidden=None):
    """Helper to handle the different architectures during inference."""
    raise NotImplementedError()


@get_next_logits.register(LSTMSeq2Seq)
def _(model, enc_inputs, dec_input, lstm_hidden=None):
    if lstm_hidden is None:
        # First step: get encoder context using packing
        src_lengths = (enc_inputs != PAD_IDX).sum(dim=1).cpu()
        packed_src = nn.utils.rnn.pack_padded_sequence(
            model.embedding(enc_inputs), src_lengths, batch_first=True, enforce_sorted=False
        )
        _, lstm_hidden = model.encoder(packed_src)
    
    # Only pass the LAST generated token to the LSTM decoder to update the state
    out, lstm_hidden = model.decoder(model.embedding(dec_input[:, -1:]), lstm_hidden)
    logits = model.fc_out(out)
    return logits[:, -1, :], lstm_hidden


@get_next_logits.register(FullTransformer)
def _(model, enc_inputs, dec_input, lstm_hidden=None):
    logits = model(enc_inputs, dec_input)
    return logits[:, -1, :], None


@get_next_logits.register(DecoderOnlyTransformer)
@get_next_logits.register(RoPEDecoderOnlyTransformer)
def _(model, enc_inputs, dec_input, lstm_hidden=None):
    combined_input = dec_input
    logits = model(combined_input)
    return logits[:, -1, :], None

