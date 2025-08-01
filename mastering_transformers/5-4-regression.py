from transformers import ( DistilBertConfig, 
                           DistilBertTokenizerFast, 
                           DistilBertForSequenceClassification, 
                           TrainingArguments, 
                           Trainer )

import datasets
from datasets import load_dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from torch import cuda
import torch

device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
print(f"**Using {device} device**")

MODEL_PATH='distilbert-base-uncased'
config = DistilBertConfig.from_pretrained(MODEL_PATH, num_labels=1)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

stsb_train= load_dataset('glue','stsb', split="train")
stsb_validation = load_dataset('glue','stsb', split="validation")
stsb_validation=stsb_validation.shuffle(seed=42)
stsb_val= datasets.Dataset.from_dict(stsb_validation[:750])
stsb_test= datasets.Dataset.from_dict(stsb_validation[750:])

enc_train = stsb_train.map(lambda e: tokenizer( e['sentence1'],e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_val =   stsb_val.map(lambda e: tokenizer( e['sentence1'],e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_test =  stsb_test.map(lambda e: tokenizer( e['sentence1'],e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000)  

training_args = TrainingArguments(
    no_cuda=False,
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./stsb-model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    # TensorBoard log directory
    logging_strategy='steps',                
    logging_dir='./logs',            
    logging_steps=50,
    # other options : no, steps
    evaluation_strategy="steps",
    save_strategy="steps",
    fp16=cuda.is_available(),
    load_best_model_at_end=True
)

def compute_metrics(pred):
    preds = np.squeeze(pred.predictions) 
    return {"MSE": ((preds - pred.label_ids) ** 2).mean().item(),
            "RMSE": (np.sqrt ((  (preds - pred.label_ids) ** 2).mean())).item(),
            "MAE": (np.abs(preds - pred.label_ids)).mean().item(),
            "Pearson" : pearsonr(preds,pred.label_ids)[0],
            "Spearman's Rank" : spearmanr(preds,pred.label_ids)[0]
            }

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
)

model.to(device)

train_result = trainer.train()
metrics = train_result.metrics


model.eval()

s1,s2="A plane is taking off.",	"An air plane is taking off."

encoding = tokenizer(s1,s2, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad():  # 추론 시에는 gradient 계산 비활성화
    outputs = model(input_ids, attention_mask=attention_mask)
print(s1, " & ", s2)
print(outputs.logits.item())

s1,s2="The men are playing soccer.",	"A man is riding a motorcycle."

encoding = tokenizer("hey how are you there","hey how are you", return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
outputs = model(input_ids, attention_mask=attention_mask)

with torch.no_grad():  # 추론 시에는 gradient 계산 비활성화
    outputs = model(input_ids, attention_mask=attention_mask)
print(outputs.logits.item())

q=[trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:6]

model_path = "sentence-pair-regression-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

