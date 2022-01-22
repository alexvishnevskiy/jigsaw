from transformers import AutoTokenizer
from box import Box
import yaml


def load_cfg(filepath):
    cfg = yaml.load(open(filepath, 'r'), Loader = yaml.Loader)
    cfg = Box(cfg)
    cfg['tokenizer'] = AutoTokenizer.from_pretrained(cfg['model_name'])
    return cfg
