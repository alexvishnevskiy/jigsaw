from sklearn.feature_extraction.text import TfidfVectorizer
from ...utils.glove import convert_glove_to_features, load_glove
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
    def __init__(self, cfg):
        self.cfg = cfg

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
        hash_X = (
            ' '.join(list(X)) + self.cfg.emb_type + \
            self.cfg.get('emb_path') if self.cfg.get('emb_path') is not None else ''
            ).encode()
        hash_X = hashlib.sha256(hash_X).hexdigest()
        cache_dir = os.path.join(Path(__file__).parents[3], '.cache')
        filename = os.path.join(cache_dir, f'{hash_X}.npy')

        if os.path.exists(filename):
            print('loading features from cache')
            features = np.load(filename)
        elif self.cfg.cache_features:
            print(f'caching features to {filename}')
            if not os.path.exists(cache_dir): 
                os.mkdir(cache_dir)

            if self.cfg.emb_type == 'tfidf':
                features = self._convert_text_to_tfidf(X)
                return features
            if self.cfg.emb_type == 'glove':
                emmbed_dict = load_glove(self.cfg.emb_path)
                features = self._convert_glove_to_features(X, emmbed_dict)
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
    
    def _convert_glove_to_features(self, X, emmbed_dict):
        return convert_glove_to_features(X, emmbed_dict)

    def _convert_fasttext_to_features(self, X):
        return convert_fasttext_to_features(X, self.cfg.emb_path, self.cfg.get('tokenizer'))

    def save(self, path):
        path = Path(path)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent)

        if self.cfg.emb_type == 'tfidf':
            #dump tfidf
            tfidf_save_path = path.with_name(f"{path.stem}-tfidf.joblib")
            joblib.dump(self.vectorizer, tfidf_save_path)
            #delete tfidf in order to clean memory
            delattr(self, 'vectorizer')
        joblib.dump(self, path) 
        return path.parent

    @classmethod
    def load(self, path):
        path = Path(path)
        cls = joblib.load(path)
        try:
            tfidf_path = path.with_name(f"{path.stem}-tfidf.joblib")
            cls.vectorizer = joblib.load(tfidf_path)
        except FileNotFoundError:
            pass
        return cls


class LinearModel(BaseModel, Ridge):
    def __init__(self, cfg, alpha=1, random_state=None):
        BaseModel.__init__(self, cfg)
        Ridge.__init__(self, alpha=alpha, random_state=random_state)


class KernelModel(BaseModel, KernelRidge):
    def __init__(self, cfg, alpha=1, kernel='linear', gamma=None, degree=3):
        BaseModel.__init__(self, cfg)
        KernelRidge.__init__(self, alpha=alpha, kernel=kernel, gamma=gamma, degree=degree)


class SVRModel(BaseModel, SVR):
    def __init__(self, cfg, kernel='rbf', degree=3, gamma='scale', C=1):
        BaseModel.__init__(self, cfg)
        SVR.__init__(self, kernel=kernel, degree=degree, gamma=gamma, C=C)
