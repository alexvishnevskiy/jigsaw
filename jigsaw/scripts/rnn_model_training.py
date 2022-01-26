from ..utils.sampler import BySequenceLengthRegressionSampler, BySequenceLengthPairedSampler
from pytorch_lightning.utilities.seed import seed_everything
from ..models.rnn_models.lightning_models import PairedRnnModel, RegressionRnnModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import callbacks
import pytorch_lightning as pl
import wandb
import os


def train(cfg, train_df, val_df):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(cfg.tokenizer, cfg.dataset.text_col, train_df)
        model = RegressionRnnModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedRnnModel(cfg, train_df, val_df)

    earystopping = EarlyStopping(monitor="val_acc", patience = 3)
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath = os.path.join(cfg.logger.save_dir, cfg.rnn_type, cfg.dataset.name),
        filename=f"{cfg.rnn_type}" if cfg.get("fold") is None \
                 else f"{cfg.rnn_type}_{cfg.get('fold')}",
        monitor="val_acc",
        save_weights_only=True,
        save_top_k=1,
        mode="max",
        save_last=False,
        )
    wandb_logger = WandbLogger(
        log_model = True,
        )

    wandb.init(project = cfg.logger.project, name = f'{cfg.rnn_type}_{cfg.dataset.name}')
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("val_loss", summary="min")

    trainer = pl.Trainer(
      max_epochs=cfg.epoch,
      logger = wandb_logger,
      callbacks=[
            lr_monitor, 
            loss_checkpoint, 
            earystopping
            ],
      precision = 16,
      #because of the sampler
      limit_train_batches = sum(1 for _ in sampler) if cfg.bucket_seq  else len(model.train_dataloader()), 
      **cfg.trainer,
      )
    trainer.fit(model)