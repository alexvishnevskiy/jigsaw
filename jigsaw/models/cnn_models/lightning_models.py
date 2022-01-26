from ..base_module import RegressionModel, PairedModel
from .base_model import CnnModel
import torch.nn as nn


class RegressionCnnModel(RegressionModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, CnnModel(cfg), train_df, val_df, test_df)

  def forward(self, input, attn_mask):
    output = self.model(input, attn_mask).squeeze()
    return output

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    output = self(batch['input_ids'], batch['attention_mask'])

    loss = self.criterion(y, output)
    self.log('train_loss', loss)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(batch['more_toxic_ids'], batch['more_toxic_mask'])
    output2 = self(batch['less_toxic_ids'], batch['less_toxic_mask'])

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    output = self(batch['input_ids'], batch['attention_mask']).squeeze().cpu()
    return output


class PairedCnnModel(PairedModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, CnnModel(cfg), train_df, val_df, test_df)

  def forward(self, input, attn_mask):
    output = self.model(input, attn_mask).squeeze()
    return output

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(batch['more_toxic_ids'], batch['more_toxic_mask'])
    output2 = self(batch['less_toxic_ids'], batch['less_toxic_mask'])

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return {'loss': loss, 'acc': acc}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(batch['more_toxic_ids'], batch['more_toxic_mask'])
    output2 = self(batch['less_toxic_ids'], batch['less_toxic_mask'])

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    output = self(batch['input_ids'], batch['attention_mask']).squeeze().cpu()
    return output