from ..utils.callbacks import CsvWritter
import pytorch_lightning as pl
from ..models import *
import pandas as pd
import torch
import os


def base_predict(cfg, model, checkpoint_path, fold = None):
  model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

  csv_writer = CsvWritter(cfg, fold=fold)
  trainer = pl.Trainer(
    gpus = 1,
    callbacks=[csv_writer],
    )
  trainer.predict(model)
  print("Prediction's done")

def deep_predict(cfg, test_df, checkpoint_path, fold = None):
  if cfg.model_type == 'paired':
    model = PairedDeepModel(cfg, test_df=test_df)
  else:
    model = RegressionDeepModel(cfg, test_df=test_df)
  base_predict(cfg, model, checkpoint_path, fold)

def rnn_predict(cfg, test_df, checkpoint_path, fold = None):
  if cfg.model_type == 'paired':
    model = PairedRnnModel(cfg, test_df=test_df)
  else:
    model = RegressionRnnModel(cfg, test_df=test_df)
  base_predict(cfg, model, checkpoint_path, fold)

def cnn_predict(cfg, test_df, checkpoint_path, fold = None):
  if cfg.model_type == 'paired':
    model = PairedCnnModel(cfg, test_df=test_df)
  else:
    model = RegressionCnnModel(cfg, test_df=test_df)
  base_predict(cfg, model, checkpoint_path, fold)

def linear_predict(cfg, test_df, checkpoint_path, fold = None):
  if not os.path.exists(cfg.output_dir):
      os.mkdir(cfg.output_dir)

  if cfg.model_type == 'linear':
      model = LinearModel.load(checkpoint_path)
  if cfg.model_type == 'kernel':
      model = KernelModel.load(checkpoint_path)
  if cfg.model_type == 'svr':
      model = SVRModel.load(checkpoint_path)

  if fold is not None:
    output_path = os.path.join(cfg.output_dir, f'submission_{fold}.csv')
  else:
    output_path = os.path.join(cfg.output_dir, f'submission.csv')
    
  df = pd.read_csv(cfg.sample_submission)
  df['score'] = model.predict(test_df[cfg.dataset.text_col])
  df.to_csv(output_path, index = False)