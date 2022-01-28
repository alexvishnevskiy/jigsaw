from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
import pickle
import nltk
import re
import os
nltk.download('punkt')


def load_glove(glove_path):
    emmbed_dict = {}
    glove_path = Path(glove_path)
    dict_path = glove_path.with_name(f'{glove_path.stem}.pickle')

    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            glove_dict = pickle.load(f)
        return glove_dict

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

    with open(dict_path, 'wb') as f:
        pickle.dump(emmbed_dict, f)
    return emmbed_dict

def convert_glove_to_features(texts, emmbed_dict):
    def mean_vectorizer(text):
        tokens = word_tokenize(text)
        len_tokens = len(tokens)
        mean_vector = np.zeros(emb_size)
        
        for token in tokens:
            try:
                mean_vector += emmbed_dict[token]
            except:
                pass
            
        mean_vector /= len_tokens
        return mean_vector

    emb_size = len(list(emmbed_dict.values())[0])
    features = tqdm(map(lambda x: mean_vectorizer(x), texts), total=len(texts), desc='converting glove to features')
    features = np.vstack(list(features))
    return features