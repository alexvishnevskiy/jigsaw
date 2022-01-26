from cgi import test
from ..base_module import RegressionModel, PairedModel
from .base_model import RnnModel
import torch.nn as nn


class RegressionRnnModel(RegressionModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, RnnModel(cfg), train_df, val_df, test_df)

  def forward(self, input, lengths):
    output = self.model(input, lengths).squeeze()
    return output

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    lengths = batch['attention_mask'].sum(axis = 1).cpu()
    output = self(batch['input_ids'], lengths)

    loss = self.criterion(y, output)
    self.log('train_loss', loss)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    more_toxic_lengths = batch['more_toxic_mask'].sum(axis = 1).cpu()
    less_toxic_lengths = batch['less_toxic_mask'].sum(axis = 1).cpu()
    output1 = self(batch['more_toxic_ids'], more_toxic_lengths)
    output2 = self(batch['less_toxic_ids'], less_toxic_lengths)

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    lengths = batch['attention_mask'].sum(axis = 1).cpu()
    output = self(batch['input_ids'], lengths).squeeze().cpu()
    return output


class PairedRnnModel(PairedModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, RnnModel(cfg), train_df, val_df, test_df)

  def forward(self, input, lengths):
    output = self.model(input, lengths).squeeze()
    return output

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    more_toxic_lengths = batch['more_toxic_mask'].sum(axis = 1).cpu()
    less_toxic_lengths = batch['less_toxic_mask'].sum(axis = 1).cpu()
    output1 = self(batch['more_toxic_ids'], more_toxic_lengths)
    output2 = self(batch['less_toxic_ids'], less_toxic_lengths)

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return {'loss': loss, 'acc': acc}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    more_toxic_lengths = batch['more_toxic_mask'].sum(axis = 1).cpu()
    less_toxic_lengths = batch['less_toxic_mask'].sum(axis = 1).cpu()
    output1 = self(batch['more_toxic_ids'], more_toxic_lengths)
    output2 = self(batch['less_toxic_ids'], less_toxic_lengths)

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    lengths = batch['attention_mask'].sum(axis = 1).cpu()
    output = self(batch['input_ids'], lengths).squeeze().cpu()
    return output
