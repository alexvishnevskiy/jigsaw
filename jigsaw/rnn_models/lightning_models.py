from ..deep_models.lightning_models import RegressionModel, PairedModel
from .base_model import RnnModel
import torch.nn as nn


class RegressionRnnModel(RegressionModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.model = RnnModel(cfg)
    self.criterion = nn.MSELoss() #L1
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'test_df', 'model', 'criterion'])

  def forward(self, x):
    return

  def train_dataloader(self):
    return

  def val_dataloader(self):
    return 

  def predict_dataloader(self):
    return 

  def training_step(self, batch, batch_idx):
    return

  def validation_step(self, batch, batch_idx):
    return

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    return

  def validation_epoch_end(self, outputs):
    return


class PairedRnnModel(PairedModel):
  def __init__(self, cfg, train_df = None, val_df = None, test_df = None):
    self.cfg = cfg
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.model = RnnModel(cfg)
    self.criterion = nn.MarginRankingLoss(margin=cfg['margin'])
    self.save_hyperparameters(cfg, ignore = ['train_df', 'val_df', 'test_df', 'model', 'criterion'])

  def forward(self, x):
    return

  def train_dataloader(self):
    return

  def val_dataloader(self):
    return 

  def predict_dataloader(self):
    return 

  def training_step(self, batch, batch_idx):
    return

  def validation_step(self, batch, batch_idx):
    return

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    return

  def validation_epoch_end(self, outputs):
    return