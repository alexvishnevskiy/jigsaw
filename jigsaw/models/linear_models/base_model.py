from sklearn.feature_extraction.text import TfidfVectorizer
from ...utils.glove import convert_glove_to_features
from ...utils.fast_text import convert_fasttext_to_features
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from pathlib import Path
import hashlib
import numpy as np
import joblib
import os


class BaseModel(BaseEstimator):
    def __init__(self, cfg, emmbed_dict = None):
        self.cfg = cfg
        self.emmbed_dict = emmbed_dict

    def fit(self, X, y):
        X = self._convert_to_features(X)
        super().fit(X, y)

    def predict(self, X):
        if self.cfg.emb_type == 'tfidf':
            X = self.vectorizer.transform(X)
        else:
            X = self._convert_to_features(X)
        return super().predict(X)

    def _convert_to_features(self, X):
        hash_X = hashlib.sha256((' '.join(list(X)) + self.cfg.emb_type).encode()).hexdigest()
        cache_dir = os.path.join(Path(__file__).parents[3], '.cache')
        filename = os.path.join(cache_dir, f'{hash_X}.pkl')

        if os.path.exists(filename):
            print('loading features from cache')
            features = np.load(filename)
        else:
            print(f'caching features to {filename}')
            if not os.path.exists(cache_dir): 
                os.mkdir(cache_dir)

            if self.cfg.emb_type == 'tfidf':
                features = self._convert_text_to_tfidf(X)
                return features
            if self.cfg.emb_type == 'glove':
                features = self._convert_glove_to_features(X)
            if self.cfg.emb_type == 'fasttext':
                features = self._convert_fasttext_to_features(X)
            #cache features
            np.save(filename, features)
        return features
    
    def _convert_text_to_tfidf(self, X):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb' if self.cfg.get('tokenizer') is not None else 'word', 
            ngram_range=self.cfg.ngram_range,
            max_df=self.cfg.max_df, min_df=self.cfg.min_df
            )
        features = self.vectorizer.fit_transform(X)
        return features
    
    def _convert_glove_to_features(self, X):
        return convert_glove_to_features(X, self.emmbed_dict)

    def _convert_fasttext_to_features(self, X):
        return convert_fasttext_to_features(X, self.cfg.emb_path, self.cfg.get('tokenizer'))

    def save(self, path):
        path = Path(path)
        if not os.path.exists(path):
            os.makedirs(path.parent)
        joblib.dump(self, path) 
        return path.parent

    @classmethod
    def load(self, path):
        return joblib.load(path)


class LinearModel(BaseModel, Ridge):
    def __init__(self, cfg, emmbed_dict = None, alpha=1, random_state=None):
        BaseModel.__init__(self, cfg, emmbed_dict)
        Ridge.__init__(self, alpha=alpha, random_state=random_state)


class KernelModel(BaseModel, KernelRidge):
    def __init__(self, cfg, emmbed_dict = None, alpha=1, kernel='linear', gamma=None, degree=3):
        BaseModel.__init__(self, cfg, emmbed_dict)
        KernelRidge.__init__(self, alpha=alpha, kernel=kernel, gamma=gamma, degree=degree)


class SVRModel(BaseModel, SVR):
    def __init__(self, cfg, emmbed_dict = None, kernel='rbf', degree=3, gamma='scale', C=1):
        BaseModel.__init__(self, cfg, emmbed_dict)
        SVR.__init__(self, kernel=kernel, degree=degree, gamma=gamma, C=C)
