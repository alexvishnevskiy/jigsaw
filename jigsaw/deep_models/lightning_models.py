from pytorch_lightning import LightningModule
from .base_model import JigsawModel
from ..datasets import (
    RegressionDataset, PairedDataset, 
    get_regression_loader, get_paired_loader
)
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch


class RegressionModel(LightningModule):
  def __init__(self, cfg, train_df, val_df):
    super().__init__()
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.model = JigsawModel(cfg)
    self.criterion = nn.MSELoss()
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'model', 'criterion'])
    
  def forward(self, input_ids = None, attention_mask = None):
    out = self.model(input_ids = input_ids, attention_mask = attention_mask)
    out = out.squeeze()
    return out

  def train_dataloader(self):
    train_split = RegressionDataset(
          self.train_df,
          self.cfg,
          self.cfg.tokenizer,
          self.cfg.dataset.text_col,
          self.cfg.dataset.target_col
      )
    loader = get_regression_loader(train_split, self.cfg.tokenizer, self.cfg.batch_size, shuffle=True)
    return loader

  def val_dataloader(self):
    val_split = PairedDataset(
        self.val_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(val_split, self.cfg.batch_size, shuffle = False)
    return loader

  def configure_optimizers(self):
    optimizer = eval(self.cfg.optimizer.name)(
        self.parameters(), **self.cfg.optimizer.params
        )
    scheduler = eval(self.cfg.scheduler.name)(
        optimizer,
        num_training_steps = int(
            len(self.train_dataloader())*
            self.cfg.epoch/self.cfg.acc_step
        ),
        **self.cfg.scheduler.params
        )
    
    scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': self.cfg.acc_step
        }
    return [optimizer], [scheduler]

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

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

    self.log('val_loss', avg_loss)
    self.log('val_acc', avg_acc)

class PairedModel(LightningModule):
  def __init__(self, cfg, train_df, val_df):
    super().__init__()
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.model = JigsawModel(cfg)
    self.criterion = nn.MarginRankingLoss(margin=cfg['margin'])
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'model', 'criterion'])
    
  def forward(self, input_ids = None, attention_mask = None):
    out = self.model(input_ids = input_ids, attention_mask = attention_mask)
    return out

  def train_dataloader(self):
    train_split = PairedDataset(
        self.train_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(train_split, self.cfg.batch_size, shuffle = True)
    return loader

  def val_dataloader(self):
    val_split = PairedDataset(
        self.train_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(val_split, self.cfg.batch_size, shuffle = False)
    return loader

  def configure_optimizers(self):
    optimizer = eval(self.cfg.optimizer.name)(
        self.parameters(), **self.cfg.optimizer.params
        )
    scheduler = eval(self.cfg.scheduler.name)(
        optimizer,
        num_training_steps = int(
            len(self.train_dataloader())*
            self.cfg.epoch/self.cfg.acc_step
        ),
        **self.cfg.scheduler.params
        )
    
    scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': self.cfg.acc_step
        }
    return [optimizer], [scheduler]

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

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

    self.log('val_loss', avg_loss)
    self.log('val_acc', avg_acc)
