from .base_model import JigsawModel
from ..base_module import RegressionModel, PairedModel
import torch.nn as nn


class RegressionDeepModel(RegressionModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, JigsawModel(cfg), train_df, val_df, test_df)
    
  def forward(self, input_ids = None, attention_mask = None):
    out = self.model(input_ids = input_ids, attention_mask = attention_mask)
    out = out.squeeze()
    return out

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    output = self(**batch)

    loss = self.criterion(y, output)
    self.log('train_loss', loss)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(input_ids = batch['more_toxic_ids'], attention_mask = batch['more_toxic_mask'])
    output2 = self(input_ids = batch['less_toxic_ids'], attention_mask = batch['less_toxic_mask'])

    loss = nn.MarginRankingLoss(margin = self.cfg['margin'])(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    output = self(**batch).squeeze().cpu()
    return output


class PairedDeepModel(PairedModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    super().__init__(cfg, JigsawModel(cfg), train_df, val_df, test_df)
    
  def forward(self, input_ids = None, attention_mask = None):
    out = self.model(input_ids = input_ids, attention_mask = attention_mask)
    return out

  def training_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(input_ids = batch['more_toxic_ids'], attention_mask = batch['more_toxic_mask'])
    output2 = self(input_ids = batch['less_toxic_ids'], attention_mask = batch['less_toxic_mask'])

    loss = self.criterion(output1, output2, y)
    acc = (output1 > output2).float().mean()
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return {'loss': loss, 'acc': acc}

  def validation_step(self, batch, batch_idx):
    y = batch.pop('target')
    output1 = self(input_ids = batch['more_toxic_ids'], attention_mask = batch['more_toxic_mask'])
    output2 = self(input_ids = batch['less_toxic_ids'], attention_mask = batch['less_toxic_mask'])

    loss = self.criterion(output1, output2, y)
    acc = (output1 > output2).float().mean()
    return {'loss': loss, 'acc': acc}

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    output = self(**batch).squeeze().cpu()
    return output
