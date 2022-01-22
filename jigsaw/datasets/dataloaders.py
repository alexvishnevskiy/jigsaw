from unittest.util import _MAX_LENGTH
from ..utils.sampler import BySequenceLengthRegressionSampler, BySequenceLengthPairedSampler
from jigsaw.utils.optimal_lenght import find_optimal_lenght
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from collections import defaultdict
from functools import partial
import torch


def collate_paired(batch):
  max_more_length = max(len(b['more_toxic_ids']) for b in batch)
  max_less_length = max(len(b['less_toxic_ids']) for b in batch)

  output = defaultdict(list)
  for b in batch:
    for k, v in b.items():
      max_length = max_more_length if 'more' in k else max_less_length
      if 'mask' in k:
        output[k].append(v + [0]*(max_length-len(v)))
      elif 'ids' in k:
        output[k].append(v + [1]*(max_length-len(v)))
      else:
        output[k].append(v)

  for k, v in output.items():
    if 'mask' in k:
      output[k] = torch.tensor(v, dtype = torch.float)
    else:
      output[k] = torch.tensor(v, dtype = torch.long)
  return output

def collate_regression(batch, tokenizer):
  collator = DataCollatorWithPadding(
      tokenizer = tokenizer,
      padding = 'longest',
      return_tensors = 'pt'
      )
  output = collator(batch)

  for k, v in output.items():
    if 'mask' in k or 'target' in k:
      output[k] = torch.tensor(v, dtype = torch.float)
    else:
      output[k] = torch.tensor(v, dtype = torch.long)
  return output

def get_paired_loader(dataset, batch_size, bucket_seq = True, shuffle = True):
    if bucket_seq:
        max_length1 = find_optimal_lenght(
            dataset.df, dataset.tokenizer, 
            dataset.more_toxic, dataset.cfg.max_length
            )
        max_length2 = find_optimal_lenght(
            dataset.df, dataset.tokenizer, 
            dataset.less_toxic, dataset.cfg.max_length
            )

        sampler = BySequenceLengthPairedSampler(
            toknizer=dataset.cfg.tokenizer, 
            text1_col=dataset.more_toxic, 
            text2_col=dataset.less_toxic, 
            data_source=dataset.df,
            max_length1=max_length1,
            max_length2=max_length2,
            batch_size=batch_size
            )
        return DataLoader(
            dataset, 
            batch_size=1,
            batch_sampler=sampler, 
            collate_fn=collate_paired,
            num_workers=4
            )
    else:
        return DataLoader(
            dataset,
            collate_fn = collate_paired,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = 4
        )

def get_regression_loader(dataset, tokenizer, batch_size, bucket_seq = True, shuffle = True):
    if bucket_seq:
        max_length = find_optimal_lenght(
            dataset.df, dataset.cfg.tokenizer, 
            dataset.cfg.dataset.text_col, dataset.cfg.max_length
            )
        sampler = BySequenceLengthRegressionSampler(
            data_source=dataset.df,
            tokenizer=dataset.cfg.tokenizer, 
            text_col=dataset.cfg.dataset.text_col,
            max_length=max_length, 
            batch_size=batch_size
            )

        return DataLoader(
            dataset, 
            batch_size=1,
            batch_sampler=sampler, 
            collate_fn=partial(collate_regression, tokenizer = tokenizer),
            num_workers=4
            )
    else:
        return DataLoader(
            dataset,
            collate_fn = partial(collate_regression, tokenizer = tokenizer),
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = 4
        )
