import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import numpy as np


class Evaluator():
  def __init__(self,model,testDataLoader):
    self.model = model
    self.testDataLoader = testDataLoader
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device_count = torch.cuda.device_count()
    if self.device_count > 1:
      print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
      self.model = torch.nn.DataParallel(self.model)
    else:
      print(f"Using {torch.cuda.device_count()} GPUs")
      self.model.to(self.device)
  def evaluate(self):
    self.model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(self.testDataLoader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    pred_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")





class Pipeline(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super(Pipeline, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        self.device_count = torch.cuda.device_count()
        if self.device_count > 1:
            print(f"Using {self.device_count} GPUs via DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f"Using {self.device_count} GPU(s)")

        self.model.to(self.device)

    def forward(self, texts):
        if not isinstance(texts, list):
            raise ValueError("Input texts should be a list of strings.")


        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']


        batch_size = input_ids.size(0)
        remainder = batch_size % self.device_count

        if remainder != 0:
            padding_size = self.device_count - remainder

            padding_input_ids = torch.zeros((padding_size, input_ids.size(1)), dtype=input_ids.dtype)
            padding_attention_mask = torch.zeros((padding_size, attention_mask.size(1)), dtype=attention_mask.dtype)

            input_ids = torch.cat([input_ids, padding_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, padding_attention_mask], dim=0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if remainder != 0:
            logits = logits[:batch_size]
        pred_labels = torch.argmax(logits, dim=1).tolist()
        return [self.label_map[label] for label in pred_labels]


