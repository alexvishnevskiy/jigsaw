from pytorch_lightning import callbacks
import pandas as pd
import torch
import os


class CsvWritter(callbacks.BasePredictionWriter):
  def __init__(self, cfg, write_interval='epoch'):
    super().__init__(write_interval)
    self.cfg = cfg
    self.configure_output_dir()
        
  def configure_output_dir(self):
    if not os.path.exists(self.cfg.output_dir):
        os.mkdir(self.cfg.output_dir)
        
  def write_on_epoch_end(
      self, trainer, pl_module, predictions, batch_indices
  ):
      df = pd.read_csv(self.cfg.sample_submission)
      predictions = torch.cat([pr.reshape(1) if pr.shape == () else pr for pr in predictions[0]])
      predictions = predictions.numpy()
     
      df['score'] = predictions
      #rankdata( predictions, method='ordinal') 
      df.to_csv(os.path.join(self.cfg.output_dir, f'submission.csv'), index = False)
      print("prediction's done")