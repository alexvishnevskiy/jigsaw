from nltk.tokenize import word_tokenize
from typing import List, Dict, Any
from collections import Counter
import pickle
import numpy as np
import torch


class Tokenizer:
    """
    tokenizer for fasttext, glove, word_tokenize
    """
    def __init__(self, tokens_dict=None):
        self.__tokens_dict = tokens_dict

    def encode_plus(
        self, 
        text,
        max_length=256,
        **kwargs
        ):
        assert self.__tokens_dict is not None, "tokenizer should be trained"

        tokens = word_tokenize(text)
        input_ids = []
        for token in tokens[:max_length]:
            if token in self.__tokens_dict:
                input_ids.append(self.__tokens_dict[token])
            else:
                input_ids.append(self.__tokens_dict['<unk>'])

        return {
            'input_ids': input_ids,
            'attention_mask': [1]*len(input_ids)
        }

    def fit(self, df, n_tokens=50_000):
        all_tokens = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                tokens = df[col].apply(lambda x: word_tokenize(x)).values.tolist()
                for tok in tokens:
                    all_tokens.extend(tok)
                    
        c = Counter(all_tokens)
        tokens_dict = {'<unk>': 0, '<pad>': 1}
        top_tokens = list(map(lambda x: x[0], c.most_common(n_tokens-2)))
        tokens_dict.update(dict(zip(top_tokens, range(2, len(top_tokens)+2))))
        self.__tokens_dict = tokens_dict

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__tokens_dict, f)

    def pad(
        self, 
        features:List[Dict[str, Any]], 
        padding=True, 
        max_length=None, 
        pad_to_multiple_of=None, 
        return_tensors="pt"
        ):
        assert len(features) > 0
        feat_keys = list(features[0].keys())

        if padding == True or padding == 'longest':
            max_length = max(len(feat[feat_keys[0]]) for feat in features)
        if padding == 'max_length':
            max_length = max_length if max_length is not None else 256
        if padding == False or padding == 'do_not_pad':
            max_length = 0

        output_dict = dict(zip(feat_keys, [[] for _ in range(len(feat_keys))]))
        for i in range(len(features)):
            for k in feat_keys:
                if k == 'target':
                    output_dict[k].append(features[i][k])
                    continue

                len_feat_k = len(features[i][k])
                max_length_sec = max_length if max_length != 0 else len(features[i][k])
                if k == 'attention_mask':
                    output_dict[k].append(features[i][k][:max_length_sec] + [0]*max(0, max_length-len_feat_k))
                else:
                    output_dict[k].append(features[i][k][:max_length_sec] + [1]*max(0, max_length-len_feat_k))
                
        for k, v in output_dict.items():
            if return_tensors == 'pt':
                output_dict[k] = torch.as_tensor(v)
            else:
                output_dict[k] = np.asarray(v)
        return output_dict

    @classmethod
    def from_pretrained(cls, path):
        with open(path, 'rb') as f:
            tokens_dict = pickle.load(f)
        return cls(tokens_dict)

    @property
    def vocab_size(self):
        return len(self.__tokens_dict)

    def tokenize(self, text):
        return word_tokenize(text)

    def get_vocab(self,):
        return self.__tokens_dict
