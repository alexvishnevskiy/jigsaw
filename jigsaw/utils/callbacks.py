import pytorch_lightning as pl
import pandas as pd
import torch
import os


class CsvWritter(pl.callbacks.BasePredictionWriter):
  def __init__(self, cfg, fold = None, write_interval='epoch'):
    super().__init__(write_interval)
    self.cfg = cfg
    self.fold = fold
    self.configure_output_dir()
        
  def configure_output_dir(self):
    if not os.path.exists(self.cfg.output_dir):
        os.mkdir(self.cfg.output_dir)
        
  def write_on_epoch_end(
      self, trainer, pl_module, predictions, batch_indices, fold = None
  ):
      df = pd.read_csv(self.cfg.sample_submission)
      predictions = torch.cat([pr.reshape(1) if pr.shape == () else pr for pr in predictions[0]])
      predictions = predictions.numpy()
     
      df['score'] = predictions
      #rankdata( predictions, method='ordinal')
      if self.fold is not None:
        output_path = os.path.join(self.cfg.output_dir, f'submission_{self.fold}.csv')
      else:
        output_path = os.path.join(self.cfg.output_dir, f'submission.csv')
      df.to_csv(output_path, index = False)
      print("prediction's done")

class FeatureExtractorFreezeUnfreeze(pl.callbacks.finetuning.BaseFinetuning):
  def __init__(self, unfreeze_at_epoch=1):
    super().__init__()
    self._unfreeze_at_epoch = unfreeze_at_epoch

  def freeze_before_training(self, pl_module):
    # freeze any module you want
    # Here, we are freezing `feature_extractor`
    self.freeze(pl_module.model.model)

  def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
    # When `current_epoch` is 10, feature_extractor will start training.
    if current_epoch == self._unfreeze_at_epoch:
      print('unfreezing backbone')
      self.unfreeze_and_add_param_group(
        modules=pl_module.model.model,
        optimizer=optimizer,
        train_bn=True,
      )