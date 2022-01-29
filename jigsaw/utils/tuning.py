import hydra
from ..scripts.training import *


@hydra.main(config_path='configs/optuna.yaml')
def tune(cfg):
    #train
    #get valuer of accuracy and return
    pass