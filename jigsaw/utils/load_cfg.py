from transformers import AutoTokenizer
from jigsaw.utils.tokenizer import Tokenizer
from box import Box
import yaml


def load_cfg(filepath):
    cfg = yaml.load(open(filepath, 'r'), Loader = yaml.Loader)
    cfg = Box(cfg)
    #add also custom tokenizer
    if cfg.get('tokenizer_path') is not None:
        cfg['tokenizer'] = Tokenizer.from_pretrained(cfg['tokenizer_path'])
    else:
        cfg['tokenizer'] = AutoTokenizer.from_pretrained(cfg['model_name'])
    return cfg
