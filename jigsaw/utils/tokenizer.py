from nltk.tokenize import word_tokenize


class Tokenizer:
    """
    tokenizer for fasttext, glove, word_tokenize
    """
    def __init__(self,):
        pass

    def encode_plus(
        self, 
        text,
        truncation=True,
        max_length=256,
        add_special_tokens=False
        ):
        pass

    @property
    def vocab_size(self):
        pass

    def tokenize(self, text):
        pass

    def get_vocab(self,):
        pass
