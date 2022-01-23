from ..deep_models.lightning_models import RegressionModel, PairedModel
from ..utils.callbacks import CsvWritter
import pytorch_lightning as pl
import torch


def predict(cfg, test_df, checkpoint_path):
    if cfg.model_type == 'paired':
      model = PairedModel(cfg, test_df=test_df)
    else:
      model = RegressionModel(cfg, test_df=test_df)
    model.load_state_dict(torch.load(checkpoint_path))

    csv_writer = CsvWritter(cfg)
    trainer = pl.Trainer(
      gpus = 1,
      callbacks=[csv_writer],
      deterministic=True,
      )
    trainer.predict(model)
    print("Prediction's done")