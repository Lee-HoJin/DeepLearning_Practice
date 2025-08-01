import torch
from transformers import ( DistilBertForSequenceClassification,
                           BertTokenizerFast,
                           DistilBertTokenizerFast
)
from transformers import  AdamW
import datasets
from datasets import load_dataset
import evaluate

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)



model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

sst2= load_dataset("glue","sst2")
metric = evaluate.load("glue", "sst2")

texts=list(sst2['train']['sentence'])
labels=list(sst2['train']['label'])
val_texts=list(sst2['validation']['sentence'])
val_labels=list(sst2['validation']['label'])

print("texts length: ", len(texts))

# I will take small portion
K=10000
train_dataset= MyDataset(tokenizer(texts[:K], truncation=True, padding=True), labels[:K])
val_dataset=  MyDataset(tokenizer(val_texts, truncation=True, padding=True), val_labels)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader =  DataLoader(val_dataset, batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
    model.eval()

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        predictions=outputs.logits.argmax(dim=-1)  
        metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
    eval_metric = metric.compute()

    print(f"epoch {epoch}: {eval_metric}")