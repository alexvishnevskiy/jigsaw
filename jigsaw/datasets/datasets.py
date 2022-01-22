from torch.utils.data import Dataset
from ..utils.optimal_lenght import find_optimal_lenght


class PairedDataset(Dataset):
    def __init__(
        self, 
        df, 
        cfg, 
        tokenizer, 
        more_toxic_col='more_toxic', 
        less_toxic_col='less_toxic'
        ):
        self.df = df
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.more_toxic = df[more_toxic_col].values
        self.less_toxic = df[less_toxic_col].values
        self.more_toxic_max_lenght = find_optimal_lenght(
            df, tokenizer, more_toxic_col, cfg.max_length
            )
        self.less_toxic_max_lenght = find_optimal_lenght(
            df, tokenizer, less_toxic_col, cfg.max_length
            )
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
                                more_toxic,
                                truncation=True,
                                max_length=self.more_toxic_max_lenght,
                                add_special_tokens=True,
                            )
        inputs_less_toxic = self.tokenizer.encode_plus(
                                less_toxic,
                                truncation=True,
                                max_length=self.less_toxic_max_lenght,
                                add_special_tokens=True,
                            )
        target = 1
        
        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']
        
        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']
        
        
        return {
            'more_toxic_ids': more_toxic_ids,
            'more_toxic_mask': more_toxic_mask,
            'less_toxic_ids': less_toxic_ids,
            'less_toxic_mask': less_toxic_mask,
            'target': target
        }


class RegressionDataset(Dataset):
  def __init__(self, df, cfg, tokenizer, text_col, target_col = None):
        self.df = df
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.X = df[text_col].values
        self.target_col = target_col
        self.max_lenght = find_optimal_lenght(
            df, tokenizer, text_col, cfg.max_length
            )
        if target_col is not None:
            self.y = df[target_col].values

  def __len__(self):
        return len(self.df)

  def __getitem__(self, index):
        text = self.X[index]
        if self.target_col is not None:
            target = self.y[index]

        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_lenght,
            add_special_tokens=True,
            )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        if self.target_col is not None:
            return {
                'input_ids': ids,
                'attention_mask': mask,
                'target': target
            }
        else:
            return {
                'input_ids': ids,
                'attention_mask': mask
            }