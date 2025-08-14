import torch, os
from torch import cuda
import numpy as np
# transformers
from transformers import AdapterTrainer
from transformers import (BertTokenizerFast, 
                          BertForSequenceClassification)
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

device = 'cuda' if cuda.is_available() else 'cpu'

model_path= 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# to take smaller portion 4K for train, 1K for test and 1K for validation
imdb_train= load_dataset('imdb', split="train[:2000]+train[-2000:]")
imdb_test= load_dataset('imdb', split="test[:500]+test[-500:]")
imdb_val= load_dataset('imdb', split="test[500:1000]+test[-1000:-500]")
imdb_train.shape, imdb_test.shape, imdb_val.shape

def tokenize_it(e):
  return tokenizer(e['text'], 
                   padding=True, 
                   truncation=True)

enc_train=imdb_train.map(tokenize_it, batched=True, batch_size=1000)
enc_test=imdb_test.map(tokenize_it, batched=True, batch_size=1000) 
enc_val=imdb_val.map(tokenize_it, batched=True, batch_size=1000)

training_args = TrainingArguments(
    "/tmp",
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    learning_rate=2e-4,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    warmup_steps=100,                
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True
)


def compute_acc(p):
  preds = np.argmax(p.predictions, axis=1)
  acc={"Accuracy": (preds == p.label_ids).mean()}
  return acc


from transformers import BertModelWithHeads
model = BertModelWithHeads\
    .from_pretrained(model_path)

# we add an adapter and named imdb_sentiment
model.add_adapter("imdb_sentiment")
# we add a classification head and asscociate it with added adapter
model.add_classification_head(
    "imdb_sentiment",
    num_labels=2)

# we tell the training process that added adpater will be trained!
model.train_adapter("imdb_sentiment")


# we count them in Millions
trainable_params=model.num_parameters(only_trainable=True)/(2**20) 
all_params=model.num_parameters() /2**20
print(f"{all_params=:.2f} M\n"+
      f"{trainable_params=:.2f} M\n"+
      f"The efficiency ratio is \
      {100*trainable_params/all_params:.2f}%")



trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    compute_metrics=compute_acc,
)

trainer.train()

import pandas as pd
q=[trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]