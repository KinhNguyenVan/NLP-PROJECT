from data import UITDataset, UITDatabuilder, preprocess_vietnamese_row
from model import SequenceClassifier
from evaluator import Evaluator, Pipeline
from trainer import Trainer
from transformers import BertTokenizer, XLNetTokenizer
from datasets import load_dataset
import torch
import os

def main():
  
  dataset = load_dataset("uitnlp/vietnamese_students_feedback")
  # preprocessing data
  # Apply the preprocessing function using the map method
  train_set = dataset['train'].map(preprocess_vietnamese_row)
  test_set = dataset['test'].map(preprocess_vietnamese_row)
  valid_set = dataset['validation'].map(preprocess_vietnamese_row)

  # Hyperparameters
  BATCH_SIZE_BERT = 64
  BATCH_SIZE_XLNET = 16
  MAX_LENGTH = 128
  EPOCHS = 5
  LEARNING_RATE = 2e-5
  WEIGHT_DECAY = 1e-3
  WARM_UP_RATIO = 0.2
  num_labels = 3
  OUTPUT_DIR = "/content/checkpoint/"
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # Data for inference
  texts = ["Môn học này rất bổ ích nhưng khối lượng kiến thức khá nặng",
    "Giảng viên giảng bài khó hiểu nhưng họ có kiến thức chuyên môn vững",
    "Môn học này rất bổ ích và thú vị",
    "Giảng viên giảng bài khó hiểu và không nhiệt tình",
    "Trường có nhiều toà nhà phục vụ cho việc học"]

  print("=======Workflow for Bert model======== /n")
  trainset = UITDataset(train_set,model_tokenizer = "bert",max_length = MAX_LENGTH)
  testset = UITDataset(test_set,model_tokenizer = "bert",max_length = MAX_LENGTH)
  validset = UITDataset(valid_set,model_tokenizer = "bert",max_length = MAX_LENGTH)
  train_loader = UITDatabuilder(trainset,batch_size = BATCH_SIZE_BERT,shuffle = True).get_dataloader()
  test_loader = UITDatabuilder(testset,batch_size = BATCH_SIZE_BERT,shuffle = False).get_dataloader()
  valid_loader = UITDatabuilder(validset,batch_size = BATCH_SIZE_BERT,shuffle = False).get_dataloader()

  model_type = "bert"
  Bert_model = SequenceClassifier(model_type,num_labels)
  trainer = Trainer(Bert_model,train_loader,valid_loader,epochs = EPOCHS,lr = LEARNING_RATE,weight_decay = WEIGHT_DECAY,warmup_raito = WARM_UP_RATIO)

  trainer.train()
  evaluator_bert = Evaluator(Bert_model,test_loader)
  evaluator_bert.evaluate()

  # Inference time
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pipeline = Pipeline(Bert_model,bert_tokenizer,device)
  print(pipeline(texts))
  # Save the model
  checkpoint_path = os.path.join(OUTPUT_DIR, 'bert_model.pth')
  torch.save(Bert_model.state_dict(), checkpoint_path)






  print("=======Workflow for XLNet model======== /n")
  trainset = UITDataset(train_set,model_tokenizer = "xlnet",max_length = MAX_LENGTH)
  testset = UITDataset(test_set,model_tokenizer = "xlnet",max_length = MAX_LENGTH)
  validset = UITDataset(valid_set,model_tokenizer = "xlnet",max_length = MAX_LENGTH)
  train_loader = UITDatabuilder(trainset,batch_size = BATCH_SIZE_XLNET,shuffle = True).get_dataloader()
  test_loader = UITDatabuilder(testset,batch_size = BATCH_SIZE_XLNET,shuffle = False).get_dataloader()
  valid_loader = UITDatabuilder(validset,batch_size = BATCH_SIZE_XLNET,shuffle = False).get_dataloader()

  model_type = "xlnet"
  Xlnet_model = SequenceClassifier(model_type,num_labels)
  trainer = Trainer(Xlnet_model,train_loader,valid_loader,epochs = EPOCHS,lr = LEARNING_RATE,weight_decay = WEIGHT_DECAY,warmup_raito = WARM_UP_RATIO)
  trainer.train()
  evaluator_xlnet = Evaluator(Xlnet_model,test_loader)
  evaluator_xlnet.evaluate()

  # Inference Time
  xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pipeline = Pipeline(Xlnet_model,xlnet_tokenizer,device)
  print(pipeline(texts))
  checkpoint_path = os.path.join(OUTPUT_DIR, 'xlnet_model.pth')
  torch.save(Xlnet_model.state_dict(), checkpoint_path)



if __name__ == "__main__":
    main()