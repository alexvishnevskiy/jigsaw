from ..rnn_models.lightning_models import RegressionRnnModel, PairedRnnModel
from ..utils.callbacks import CsvWritter
import pytorch_lightning as pl
import torch


def predict(cfg, test_df, checkpoint_path, fold = None):
    if cfg.model_type == 'paired':
      model = PairedRnnModel(cfg, test_df=test_df)
    else:
      model = RegressionRnnModel(cfg, test_df=test_df)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    csv_writer = CsvWritter(cfg, fold=fold)
    trainer = pl.Trainer(
      gpus = 1,
      callbacks=[csv_writer],
      deterministic=True,
      )
    trainer.predict(model)
    print("Prediction's done")