from transformers import (
    get_linear_schedule_with_warmup
)
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import numpy as np
class Trainer():
  def __init__(self,model,trainDataLoader,valDataLoader,epochs = 3,lr = 2e-4,weight_decay = 1e-3,warmup_raito = 0.2):
    self.model = model
    self.warmup_raito = warmup_raito
    self.trainDataLoader = trainDataLoader
    self.valDataLoader = valDataLoader
    self.epochs = epochs
    self.lr = lr
    self.weight_decay = weight_decay
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.total_steps = len(self.trainDataLoader) * self.epochs
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_raito*self.total_steps, num_training_steps=self.total_steps)
    self.device_count = torch.cuda.device_count()
    self.criterion = torch.nn.CrossEntropyLoss()
    if self.device_count > 1:
      print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
      self.model = torch.nn.DataParallel(self.model)
    else:
      print(f"Using {torch.cuda.device_count()} GPUs")
      self.model.to(self.device)
  def training_step(self):
    self.model.train()
    total_loss = 0
    for batch in tqdm(self.trainDataLoader, desc="Training"):
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      labels = batch['labels'].to(self.device)

      outputs = self.model(input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      loss = self.criterion(logits,labels)
      loss.backward()
      total_loss += loss.item()
      self.optimizer.step()
      self.scheduler.step()
      self.optimizer.zero_grad()

    avg_train_loss = total_loss / len(self.trainDataLoader)
    return avg_train_loss

  def validate_step(self):
    self.model.eval()
    total_loss = 0
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(self.valDataLoader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = self.criterion(logits,labels)
            total_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

    avg_val_loss = total_loss / len(self.valDataLoader)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    pred_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')

    metrics = {"accuracy":accuracy,"f1":f1,"precision":precision,"recall":recall}
    return avg_val_loss,metrics

  def train(self):
    for epoch in range(self.epochs):
      print(f"\nEpoch {epoch + 1}/{self.epochs}")
      train_loss = self.training_step()
      val_loss,metrics = self.validate_step()

      print(f"Train Loss: {train_loss:.4f}")
      print(f"Val Loss: {val_loss:.4f}")
      print(f"Val Accuracy: {metrics['accuracy']:.4f}")
      print(f"Val F1 Score: {metrics['f1']:.4f}")
      print(f"Val Precision: {metrics['precision']:.4f}")
      print(f"Val Recall: {metrics['recall']:.4f}")
