from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
import numpy as np
import nltk
import re
nltk.download('punkt')


def load_glove(glove_path):
    emmbed_dict = {}
    with open(glove_path, 'r') as f:
        n_words = len(f.readlines())
        f.seek(0)
        for line in tqdm(f, total = n_words, desc = 'loading glove embeddings'):
            values = line.split()
            vector = list(filter(lambda x: re.match('-?\d{1}\.\d+', x), values))
            word = ''.join([v for v in values if v not in vector])
            try:
                vector = np.asarray(vector, 'float32')
                emmbed_dict[word]=vector
            except:
                pass
    return emmbed_dict

def convert_glove_to_features(texts, emmbed_dict):
    def mean_vectorizer(text):
        tokens = word_tokenize(text)
        len_tokens = len(tokens)
        mean_vector = np.zeros(len(list(emmbed_dict.values())[0]))
        
        for token in tokens:
            try:
                mean_vector += emmbed_dict[token]
            except:
                pass
            
        mean_vector /= len_tokens
        return mean_vector

    
    features = tqdm(map(lambda x: mean_vectorizer(x), texts), total=len(texts), desc='converting glove to features')
    features = np.vstack(list(features))
    return features