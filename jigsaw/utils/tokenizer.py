from nltk.tokenize import word_tokenize
from collections import Counter
import pickle


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
        tokens_dict = {'<unk>': 0}
        top_tokens = list(map(lambda x: x[0], c.most_common(n_tokens-1)))
        tokens_dict.update(dict(zip(top_tokens, range(1, len(top_tokens)+1))))
        self.__tokens_dict = tokens_dict

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__tokens_dict, f)

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
