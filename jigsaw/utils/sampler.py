from torch.utils.data import Sampler
import numpy as np
from random import shuffle
import pandas as pd



class BySequenceLengthRegressionSampler(Sampler):
    def __init__(self, tokenizer, text_col, data_source, max_length=256, batch_size=64,):
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.data_source = data_source
        self.max_length = max_length
        self.ind_n_len = self.__get_lengths(data_source)
        self.bucket_boundaries = self.__get_bucket_boundaries()
        self.batch_size = batch_size 

    def __get_lengths(self, data_source):
      ind_n_len = []
      data_source = data_source[self.text_col].apply(
          lambda x: min(len(self.tokenizer.tokenize(x)), self.max_length)
          )
      for i, p in enumerate(data_source):
            ind_n_len.append( (i, p) )
      return ind_n_len

    def __get_bucket_boundaries(self):
      lenghts = np.array(self.ind_n_len)[:, 1]
      bucket_boundaries = pd.qcut(lenghts, 8, retbins = True, duplicates = 'drop')[1]
      return bucket_boundaries
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


class BySequenceLengthPairedSampler(Sampler):
    def __init__(
        self, 
        tokenizer, 
        text1_col, 
        text2_col, 
        data_source, 
        max_length1=256,
        max_length2=256,
        batch_size=64,):
    
        self.tokenizer = tokenizer
        self.text1_col = text1_col
        self.text2_col = text2_col
        self.max_length1 = max_length1
        self.max_length2 = max_length2
        self.data_source = data_source
        self.ind_n_len = self.__get_lengths(data_source)
        self.bucket_boundaries = self.__get_bucket_boundaries()
        self.batch_size = batch_size 

    def __get_lengths(self, data_source):
      text1 = data_source[self.text1_col].apply(
          lambda x: min(len(self.tokenizer.tokenize(x)), self.max_length1)
          )
      text2 = data_source[self.text2_col].apply(
          lambda x: min(len(self.tokenizer.tokenize(x)), self.max_length2)
          )
      
      ind_n_len = []
      data_source = list(map(lambda x: max(x), zip(text1, text2)))
      for i, p in enumerate(data_source):
            ind_n_len.append( (i, p) )
      return ind_n_len

    def __get_bucket_boundaries(self):
      lenghts = np.array(self.ind_n_len)[:, 1]
      bucket_boundaries = pd.qcut(lenghts, 8, retbins = True, duplicates = 'drop')[1]
      return bucket_boundaries
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id
