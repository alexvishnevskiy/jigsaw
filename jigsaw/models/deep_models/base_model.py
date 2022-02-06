from transformers import AutoModel
import torch.nn as nn


class JigsawModel(nn.Module):
    def __init__(self, cfg):
        super(JigsawModel, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.fc = nn.LazyLinear(cfg.num_classes)
        if cfg.freeze_backbone: self.freeze_backbone()
        
    def forward(self, input_ids, attention_mask):        
        out = self.model(input_ids=input_ids,attention_mask=attention_mask,
                         output_hidden_states=True,
                         output_attentions=True
                         )
        #добавить карту аттеншн
        try:
            output = out.pooler_output # (batch_size, hidden_size)
        except:
            output = out.last_hidden_state #(batch_size, sequence_length, hidden_size)
            output = output.mean(1).squeeze()
        outputs = self.fc(output)
        return outputs

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False