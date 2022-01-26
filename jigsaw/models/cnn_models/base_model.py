from ..rnn_models.base_model import RnnModel
import torch.nn as nn
import torch


class CnnModel(nn.Module):
    def __init__(self, cfg):
        super(CnnModel, self).__init__()
        #add initialize from pretrained embeddings
        self.cfg = cfg
        if cfg.rnn_embeddings:
            self.embeddings = RnnModel(cfg)
        else:
            self.embeddings = nn.Embedding(cfg.tokenizer.vocab_size, cfg.emb_size)

        self.conv1ds = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv1d(in_channels = cfg.emb_size, out_channels = cfg.out_channels, kernel_size=k),
                nn.GELU()
             ]) for k in range(3, 6)
            ])
        self.fc = nn.LazyLinear(cfg.num_classes)

    def forward(self, x, attention_mask):
        if self.cfg.rnn_embeddings:
            lengths = attention_mask.sum(axis = 1).cpu()
            embeddings = self.embeddings(x, lengths)
        else:
            embeddings = self.embeddings(x)
        embeddings = embeddings*attention_mask.unsqueeze(-1)  
        embeddings = embeddings.permute((0, 2, 1))  #(N, C, L)
        
        output = [conv1d(embeddings) for conv1d in self.conv1ds]
        output = torch.cat(output, dim = -1) #(batch_size, 3*out_channels, K)
        output = output.permute((0, 2, 1)) #(batch_size, K, 3*out_channels)
        output = output.mean(axis = 1) #(batch_size, 1, 3*out_channels)
        output = self.fc(output.squeeze())
        return output
