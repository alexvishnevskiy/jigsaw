import numpy as np


def find_optimal_lenght(df, tokenizer, text_col, max_length = 256):
  pr_res = np.percentile(df[text_col].apply(lambda x: len(tokenizer.tokenize(x))), 90)
  res = min(pr_res, max_length)
  return res