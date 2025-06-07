from transformers import (
    BertTokenizer,
     XLNetTokenizer
)
import torch
from torch.utils.data import Dataset, DataLoader
import regex as re


class UITDataset(Dataset):
    def __init__(self, dataset, model_tokenizer: str, max_length=128):
        self.dataset = dataset
        if model_tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_tokenizer == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_length = max_length
        self.dataset = dataset
        self.labels = [data['sentiment'] for data in self.dataset]
        self.texts = [data['sentence'] for data in self.dataset]
        self.encodings = self.tokenizer(self.texts, padding="max_length", truncation=True, return_tensors="pt",max_length = self.max_length)

    def __len__(self):
      return len(self.labels)
    def __getitem__(self,idx):
      if isinstance(idx, list):
            input_ids = self.encodings['input_ids'][idx]
            attention_mask = self.encodings['attention_mask'][idx]
            labels = torch.tensor([self.labels[i] for i in idx])
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
      else:
          input_ids = self.encodings['input_ids'][idx]
          attention_mask = self.encodings['attention_mask'][idx]
          label = torch.tensor(self.labels[idx])
          return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

class UITDatabuilder():
  def __init__(self,dataset: Dataset,batch_size = 8,shuffle = False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.dataloader = DataLoader(self.dataset,batch_size = self.batch_size,shuffle = self.shuffle,drop_last=True)
  def get_dataloader(self):
    return self.dataloader
  


def removepunc(text):
    lists="""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
    for i in lists:
        text=text.replace(i," ")
    return text

def text_strip(row):

    row = removepunc(row)
    row = re.sub("(\s+)",' ',str(row)).lower()
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower()
    row = row.rstrip()
    return row

def preprocess_vietnamese_row(row):
  """
  Applies text stripping to a single row of the dataset.
  dataset: A dictionary has the format such as {'sentence': str, 'sentiment': int,'topic': int}
  return: A dictionary with the 'sentence' field preprocessed.
  """
  row['sentence'] = text_strip(row['sentence'])
  return row
