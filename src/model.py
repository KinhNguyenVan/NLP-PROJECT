from transformers import (
    BertForSequenceClassification,
    XLNetForSequenceClassification
)
import torch


class SequenceClassifier(torch.nn.Module):
  def __init__(self,model_type: str, num_labels: int):
    super(SequenceClassifier, self).__init__()
    self.num_labels = num_labels
    if model_type == 'bert':
      print("Loading bert-case-uncased model")
      self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
    elif model_type == 'xlnet':
      print("Loading xlnet-base-cased model")
      self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=self.num_labels)
    else:
            raise ValueError("Model type must be either 'bert' or 'xlnet'")
  def forward(self, input_ids,attention_mask,labels=None):
    return self.model(input_ids,attention_mask=attention_mask,labels=labels)