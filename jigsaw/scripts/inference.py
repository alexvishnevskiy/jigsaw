from ..utils.callbacks import CsvWritter
import pytorch_lightning as pl
from ..models import *
import torch


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
