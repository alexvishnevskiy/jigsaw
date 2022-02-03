from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ...utils.glove import load_glove
import fasttext
import torch.nn as nn
import torch


class RnnModel(nn.Module):
    def __init__(self, cfg):
        super(RnnModel, self).__init__()
        self.cfg = cfg
        #add initialize from pretrained embeddings
        self.emb_size = cfg.emb_size
        self.embeddings = nn.Embedding(cfg.tokenizer.vocab_size, cfg.emb_size)
        if cfg.load_embeddings: self.load_embeddings()
        if cfg.freeze_embeddings: self.freeze_embeddings()

        if cfg.rnn_type == 'gru':
            self.model = nn.GRU(
                input_size=self.emb_size, 
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional
            )
        else:
            self.model = nn.LSTM(
                input_size=self.emb_size, 
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional
            )
        self.fc = nn.LazyLinear(cfg.num_classes)

    def load_embeddings(self):
        if self.cfg.emb_type == 'fasttext':
            model = fasttext.load_model(self.cfg.emb_path)
            self.emb_size = model.get_dimension()
            self.embeddings = nn.Embedding(self.cfg.tokenizer.vocab_size, self.emb_size)
            for w, i in self.cfg.tokenizer.get_vocab().items():
                #copy embedding
                vector = torch.from_numpy(model.get_word_vector(w))
                self.embeddings.weight.data[i] = vector
        else:
            emb_dict = load_glove(self.cfg.emb_path)
            self.emb_size = len(list(emb_dict.values())[0])
            self.embeddings = nn.Embedding(self.cfg.tokenizer.vocab_size, self.emb_size)
            for w, i in self.cfg.tokenizer.get_vocab().items():
                try:
                    vector = emb_dict[w]
                    self.embeddings.weight.data[i] = vector
                except:
                    pass
    
    def freeze_embeddings(self):
        self.embeddings.weight.requires_grad = False

    def forward(self, x, lengths):
        embedded_seq_tensor = self.embeddings(x)
        packed_input = pack_padded_sequence(embedded_seq_tensor, lengths, enforce_sorted = False, batch_first=True)
        packed_output, (_) = self.model(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True) #(N,L,D*H_out)
        if not self.cfg.rnn_embeddings:
            #to avoid padding
            device = next(self.parameters()).device
            norm_tensor = torch.Tensor(lengths).unsqueeze(1).to(device)
            output = output.sum(axis = 1)/norm_tensor #(N, 1, D*H_out)
            output = self.fc(output.squeeze())
        return output
