from ..utils.sampler import BySequenceLengthRegressionSampler, BySequenceLengthPairedSampler
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from ..models import *
import wandb
import os


def base_train(cfg, model, sampler, save_name, checkpoint_args = [], callbacks = []):
    earystopping = EarlyStopping(monitor="val_acc", patience = 3)
    lr_monitor = pl.callbacks.LearningRateMonitor()
    summary_callback = pl.callbacks.ModelSummary(max_depth=2)
    loss_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath = os.path.join(cfg.logger.save_dir, save_name, cfg.dataset.name),
        filename=f"{save_name}_{'_'.join(checkpoint_args)}" if cfg.get("fold") is None \
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

    wandb.init(project = cfg.logger.project, name = f"{save_name}_{'_'.join(checkpoint_args)}_{cfg.dataset.name}")
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("val_loss", summary="min")

    trainer = pl.Trainer(
      max_epochs=cfg.epoch,
      logger = wandb_logger,
      callbacks=[
            lr_monitor, 
            loss_checkpoint, 
            earystopping,
            summary_callback,
            *callbacks
            ],
      precision = 16,
      #because of the sampler
      limit_train_batches = sum(1 for _ in sampler) if cfg.bucket_seq  else len(model.train_dataloader()), 
      **cfg.trainer,
      )
    trainer.fit(model)

def deep_train(cfg, train_df, val_df, checkpoint_args = [], callbacks = []):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(
            cfg.tokenizer, cfg.dataset.text_col, train_df, 
            max_length=cfg.max_length, batch_size=cfg.batch_size
            )
        model = RegressionDeepModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedDeepModel(cfg, train_df, val_df)

    base_train(cfg, model, sampler, cfg.model_name, checkpoint_args, callbacks)
    
def rnn_train(cfg, train_df, val_df, checkpoint_args = [], callbacks = []):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(
            cfg.tokenizer, cfg.dataset.text_col, train_df,
            max_length=cfg.max_length, batch_size=cfg.batch_size
            )
        model = RegressionRnnModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedRnnModel(cfg, train_df, val_df)

    base_train(cfg, model, sampler, cfg.rnn_type, checkpoint_args, callbacks)

def cnn_train(cfg, train_df, val_df, checkpoint_args = [], callbacks = []):
    seed_everything(cfg.seed)

    if cfg.dataset.type == 'regression':
        sampler = BySequenceLengthRegressionSampler(
            cfg.tokenizer, cfg.dataset.text_col, train_df,
            max_length=cfg.max_length, batch_size=cfg.batch_size
            )
        model = RegressionCnnModel(cfg, train_df, val_df)
    else:
        sampler = BySequenceLengthPairedSampler(
            cfg.tokenizer, cfg.dataset.more_toxic_col, cfg.dataset.less_toxic_col,
            train_df, cfg.max_length, cfg.max_length, cfg.batch_size
            )
        model = PairedCnnModel(cfg, train_df, val_df)
    
    model_name = 'cnn_rnn' if cfg.rnn_embeddings else 'cnn'
    base_train(cfg, model, sampler, model_name, checkpoint_args, callbacks)

def linear_train(cfg, train_df, val_df, checkpoint_args = []):
    seed_everything(cfg.seed)

    if cfg.model_type == 'linear':
        model = LinearModel(
            cfg, alpha=cfg.alpha, random_state=cfg.seed
            )
    if cfg.model_type == 'kernel':
        model = KernelModel(
            cfg, alpha=cfg.alpha, kernel=cfg.kernel, 
            gamma=cfg.gamma, degree=cfg.degree
            )
    if cfg.model_type == 'svr':
        model = SVRModel(
            cfg, kernel=cfg.kernel, degree=cfg.degree,
            gamma=cfg.gamma, C=cfg.C
            )

    wandb.init(config=cfg, project = cfg.project, name = f"{cfg.model_type}_{'_'.join(checkpoint_args)}_{cfg.dataset.name}")
    wandb.define_metric("val_acc", summary="max")
    #fit model
    X_train = train_df[cfg.dataset.text_col]
    y_train = train_df[cfg.dataset.target_col]
    X_more_toxic = val_df['more_toxic']
    X_less_toxic = val_df['less_toxic']

    model.fit(X_train, y_train)
    more_toxic_y = model.predict(X_more_toxic)
    less_toxic_y = model.predict(X_less_toxic)
    acc = (more_toxic_y > less_toxic_y).mean()

    trained_model_artifact = wandb.Artifact(f"{cfg.model_type}_{'_'.join(checkpoint_args)}_{cfg.dataset.name}", type=cfg.model_type)
    save_dir = model.save(cfg.save_path)
    trained_model_artifact.add_dir(save_dir)
    wandb.log_artifact(trained_model_artifact)
    wandb.log({'val_acc': acc})
    wandb.finish()
    return acc