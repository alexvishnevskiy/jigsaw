from ..deep_models.lightning_models import RegressionModel
from ..utils.callbacks import CsvWritter
import pytorch_lightning as pl


def predict(cfg, test_df):
    model = RegressionModel(cfg, test_df = test_df)
    csv_writer = CsvWritter(cfg)
    trainer = pl.Trainer(
      gpus = 1,
      callbacks=[csv_writer],
      deterministic=True,
      )
    trainer.predict(model)
    print("Prediction's done")