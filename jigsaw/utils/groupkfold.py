from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import numpy as np
import pandas as pd


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x

def get_group_unionfind(train: pd.DataFrame):
    less_unique_text = train['less_toxic'].unique()
    more_unique_text = train['more_toxic'].unique()
    unique_text = np.hstack([less_unique_text, more_unique_text])
    unique_text = np.unique(unique_text).tolist()    
    text2num = {text: i for i, text in enumerate(unique_text)}
    num2text = {num: text for text, num in text2num.items()}
    train['num_less_toxic'] = train['less_toxic'].map(text2num)
    train['num_more_toxic'] = train['more_toxic'].map(text2num)

    uf = UnionFind(len(unique_text))
    for seq1, seq2 in train[['num_less_toxic', 'num_more_toxic']].to_numpy():
        uf.union(seq1, seq2)

    text2group = {num2text[i]: uf.find(i) for i in range(len(unique_text))}
    train['group'] = train['less_toxic'].map(text2group)
    train = train.drop(columns=['num_less_toxic', 'num_more_toxic'])
    return train

def add_folds(train_df, n_splits = 5, seed = 2022):
    train_df = train_df.sample(frac=1, random_state=seed)
    train_df = get_group_unionfind(train_df)
    group_kfold = GroupKFold(n_splits=n_splits)

    for fold, (trn_idx, val_idx) in enumerate(group_kfold.split(train_df, train_df, train_df['group'])): 
        train_df.loc[val_idx , "fold"] = fold
    train_df["fold"] = train_df["fold"].astype(int)
    return train_df
