from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch


class RnnModel(nn.Module):
    def __init__(self, cfg):
        super(RnnModel, self).__init__()
        #add initialize from pretrained embeddings
        self.embeddings = nn.Embedding(cfg.tokenizer.vocab_size, cfg.emb_size)
        if cfg.rnn_type == 'gru':
            self.model = nn.GRU(
                input_size=cfg.emb_size, 
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional
            )
        else:
            self.model = nn.LSTM(
                input_size=cfg.emb_size, 
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional
            )
        self.fc = nn.LazyLinear(cfg.num_classes)

    def forward(self, x, lengths):
        embedded_seq_tensor = self.embeddings(x)
        packed_input = pack_padded_sequence(embedded_seq_tensor, lengths, enforce_sorted = False, batch_first=True)
        packed_output, (_) = self.model(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True) #(N,L,D*H_out)
        #to avoid padding
        output = output.sum(axis = 1)/torch.Tensor(lengths).unsqueeze(1) #(N, 1, D*H_out)
        output = self.fc(output.squeeze())
        return output


if __name__ == '__main__':
    from box import Box

    cfg = {
        'tokenizer': {'vocab_size': 5},
        'rnn_type': 'gru',
        'emb_size': 2,
        'hidden_size': 2,
        'num_layers': 1,
        'bidirectional': False,
        'num_classes': 1
    }
    cfg = Box(cfg)
    model = RnnModel(cfg)
    input_tensor = torch.tensor([[1, 1, 0],
                                 [2, 0, 0]], dtype = torch.long)
    lengths = [2, 1]
    print(model(input_tensor, lengths))