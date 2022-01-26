from ..utils.sampler import BySequenceLengthRegressionSampler, BySequenceLengthPairedSampler
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import callbacks
import pytorch_lightning as pl
from ..models import *
import wandb
import os


def base_train(cfg, model, sampler, save_name, train_df, val_df):
    earystopping = EarlyStopping(monitor="val_acc", patience = 3)
    lr_monitor = callbacks.LearningRateMonitor()
    summary_callback = callbacks.ModelSummary(max_depth=2)
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath = os.path.join(cfg.logger.save_dir, save_name, cfg.dataset.name),
        filename=f"{save_name}" if cfg.get("fold") is None \
                 else f"{save_name}_{cfg.get('fold')}",
        monitor="val_acc",
        save_weights_only=True,
        save_top_k=1,
        mode="max",
        save_last=False,
        )
    wandb_logger = WandbLogger(
        log_model = True,
        )

    wandb.init(project = cfg.logger.project, name = f'{save_name}_{cfg.dataset.name}')
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("val_loss", summary="min")

    trainer = pl.Trainer(
      max_epochs=cfg.epoch,
      logger = wandb_logger,
      callbacks=[
            lr_monitor, 
            loss_checkpoint, 
            earystopping,
            summary_callback
            ],
      precision = 16,
      #because of the sampler
      limit_train_batches = sum(1 for _ in sampler) if cfg.bucket_seq  else len(model.train_dataloader()), 
      **cfg.trainer,
      )
    trainer.fit(model)

def deep_train(cfg, train_df, val_df):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(cfg.tokenizer, cfg.dataset.text_col, train_df)
        model = RegressionDeepModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedDeepModel(cfg, train_df, val_df)

    base_train(cfg, model, sampler, cfg.model_name, train_df, val_df)
    
def rnn_train(cfg, train_df, val_df):
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

    base_train(cfg, model, sampler, cfg.rnn_type, train_df, val_df)

def cnn_train(cfg, train_df, val_df):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(cfg.tokenizer, cfg.dataset.text_col, train_df)
        model = RegressionCnnModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedCnnModel(cfg, train_df, val_df)
    
    model_name = 'cnn_rnn' if cfg.rnn_embeddings else 'cnn'
    base_train(cfg, model, sampler, model_name, train_df, val_df)