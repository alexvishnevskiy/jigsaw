from transformers import AutoModel
import torch.nn as nn


class JigsawModel(nn.Module):
    def __init__(self, cfg):
        super(JigsawModel, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.fc = nn.LazyLinear(cfg.num_classes)
        
    def forward(self, input_ids, attention_mask):        
        out = self.model(input_ids=input_ids,attention_mask=attention_mask,
                         output_hidden_states=True,
                         output_attentions=True
                         )
        #добавить карту аттеншн
        outputs = self.fc(out.pooler_output)
        return outputs