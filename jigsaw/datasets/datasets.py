from torch.utils.data import Dataset


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
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
                                more_toxic,
                                truncation=True,
                                max_length=self.cfg.max_length,
                                add_special_tokens=True,
                            )
        inputs_less_toxic = self.tokenizer.encode_plus(
                                less_toxic,
                                truncation=True,
                                max_length=self.cfg.max_length,
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
  def __init__(self, df, cfg, tokenizer, text_col, target_col):
        self.df = df
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.X = df[text_col].values
        self.y = df[target_col].values

  def __len__(self):
        return len(self.df)

  def __getitem__(self, index):
        text, target = self.X[index], self.y[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=True,
            )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'target': target
        }
