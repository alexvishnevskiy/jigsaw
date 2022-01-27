from nltk.tokenize import word_tokenize
# from .cleaning import *
from tqdm.notebook import tqdm
from typing import Union
import pandas as pd
import numpy as np
import fasttext
import nltk
import os
nltk.download('punkt')


#needs cleaning script
def train_fasttext(save_dir, train_path: Union[list, str], lr=0.05, dim=100, ws=5, epoch=10):
    def prepare_df(path):
        train_df = pd.read_csv(path)
        text = ''

        for col in train_df.columns:
            if train_df[col].dtype == 'object' or train_df[col].dtype == 'string':
                text += '\n'.join(train_df[col].values.tolist())
        return text

    train_path = [train_path] if isinstance(train_path, str) else train_path
    text = '\n'.join([prepare_df(path) for path in train_path])

    if not os.path.exists('./.temp'): os.mkdir('./.temp')    
    save_path = './.temp/temp_train.txt'
    with open(save_path, 'w') as f:
        f.write(text)

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    model = fasttext.train_unsupervised(
        './.temp/temp_train.txt', model="cbow", 
        lr=lr, dim=dim, ws=ws, epoch=epoch
        )
    model.save_model(os.path.join(save_dir, f'model_{dim}_{ws}.bin')) #bin path
    #remove train files
    os.remove('./.temp/temp_train.txt')
    os.rmdir('./.temp')

def convert_fasttext_to_features(texts, model_path, tokenizer=None):
    def mean_vectorizer(text):
        tokens = word_tokenize(text) if tokenizer is None else tokenizer.tokenize(text)
        len_tokens = len(tokens)
        mean_vector = np.zeros(model.get_dimension())
        
        for token in tokens:
            try:
                mean_vector += model.get_word_vector(token)
            except:
                pass
            
        mean_vector /= len_tokens
        return mean_vector
    
    model = fasttext.load_model(model_path)
    features = tqdm(
        map(lambda x: mean_vectorizer(x), texts), 
        total=len(texts), desc='converting fasttext to features'
        )
    features = np.vstack(list(features))
    return features
