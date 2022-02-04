from pytorch_lightning import LightningModule
from ..datasets import (
    RegressionDataset, PairedDataset, 
    get_regression_loader, get_paired_loader
)
from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
import math


class RegressionModel(LightningModule):
  def __init__(self, cfg, model, train_df = None, val_df = None, test_df = None):
    super().__init__()
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.model = model
    self.criterion = nn.MSELoss() #L1
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'test_df', 'model', 'criterion'])

  def train_dataloader(self):
    train_split = RegressionDataset(
          self.train_df,
          self.cfg,
          self.cfg.tokenizer,
          self.cfg.dataset.text_col,
          self.cfg.dataset.target_col
      )
    loader = get_regression_loader(
        train_split,
        tokenizer=self.cfg.tokenizer, 
        batch_size=self.cfg.batch_size, 
        bucket_seq=self.cfg.bucket_seq,
        shuffle=True
        )
    return loader

  def val_dataloader(self):
    val_split = PairedDataset(
        self.val_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(val_split, self.cfg.batch_size, bucket_seq=False, shuffle=False)
    return loader

  def predict_dataloader(self):
    test_split = RegressionDataset(
        self.test_df,
        self.cfg,
        self.cfg.tokenizer,
        self.cfg.dataset.text_col
    )
    loader = get_regression_loader(
        test_split, self.cfg.tokenizer, 
        self.cfg.batch_size, bucket_seq=False, shuffle = False
        )
    return loader

  def __apply_weight_decay(self):
    no_decay = []
    decay = []
    for n, p in self.named_parameters():
        if 'bias' in n and 'LayerNorm' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': no_decay, 'weight_decay': 0}, {'params': decay}]

  def _get_n_steps(self):
    len_loader = math.ceil(len(self.train_df)/self.cfg.batch_size)
    num_steps = int(self.cfg.epoch*len_loader/self.cfg.acc_step)
    return num_steps

  def configure_optimizers(self):
    n_steps = self._get_n_steps()

    optimizer = eval(self.cfg.optimizer.name)(
        self.__apply_weight_decay(), **self.cfg.optimizer.params
        )
    scheduler = eval(self.cfg.scheduler.name)(
        optimizer,
        num_training_steps = n_steps,
        num_warmup_steps = int(self.cfg.scheduler.params.num_warmup_steps*n_steps)
        )
    
    scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': self.cfg.acc_step
        }
    return [optimizer], [scheduler]

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

    self.log('val_loss', avg_loss)
    self.log('val_acc', avg_acc)


class PairedModel(LightningModule):
  def __init__(self, cfg, model, train_df = None, val_df = None, test_df = None):
    super().__init__()
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.model = model
    self.criterion = nn.MarginRankingLoss(margin=cfg['margin'])
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'test_df', 'model', 'criterion'])

  def train_dataloader(self):
    train_split = PairedDataset(
        self.train_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(
        train_split, 
        batch_size=self.cfg.batch_size, 
        bucket_seq=self.cfg.bucket_seq,
        shuffle = True)
    return loader

  def val_dataloader(self):
    val_split = PairedDataset(
        self.val_df, 
        self.cfg,
        self.cfg.tokenizer
    )
    loader = get_paired_loader(val_split, self.cfg.batch_size, bucket_seq=False, shuffle = False)
    return loader

  def predict_dataloader(self):
    test_split = RegressionDataset(
        self.test_df,
        self.cfg,
        self.cfg.tokenizer,
        self.cfg.dataset.text_col
    )
    loader = get_regression_loader(
        test_split, self.cfg.tokenizer, 
        self.cfg.batch_size, bucket_seq=False, shuffle = False
        )
    return loader

  def __apply_weight_decay(self):
    no_decay = []
    decay = []
    for n, p in self.named_parameters():
        if 'bias' in n and 'LayerNorm' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': no_decay, 'weight_decay': 0}, {'params': decay}]

  def _get_n_steps(self):
    len_loader = math.ceil(len(self.train_df)/self.cfg.batch_size)
    num_steps = int(self.cfg.epoch*len_loader/self.cfg.acc_step)
    return num_steps

  def configure_optimizers(self):
    n_steps = self._get_n_steps()

    optimizer = eval(self.cfg.optimizer.name)(
        self.__apply_weight_decay(), **self.cfg.optimizer.params
        )
    scheduler = eval(self.cfg.scheduler.name)(
        optimizer,
        num_training_steps = n_steps,
        num_warmup_steps = int(self.cfg.scheduler.params.num_warmup_steps*n_steps),
        )
    
    scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': self.cfg.acc_step
        }
    return [optimizer], [scheduler]

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

    self.log('val_loss', avg_loss)
    self.log('val_acc', avg_acc)
