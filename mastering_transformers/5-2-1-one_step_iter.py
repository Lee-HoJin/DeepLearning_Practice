from transformers import ( DistilBertForSequenceClassification, 
                           DistilBertTokenizerFast, 
                           AdamW
)
import torch
from torch.nn import functional

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.train()

tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')

optimizer = AdamW(model.parameters(), lr=1e-3)

texts= ["this is a good example","this is a bad example","this is a good one"]
labels= [1,0,1]
labels = torch.tensor(labels).unsqueeze(0)

encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print("###   outputs:")
print(outputs)

labels = torch.tensor([1,0,1])
outputs = model(input_ids, attention_mask=attention_mask)
loss = functional.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()
loss

print("###   manual outputs:")
print(outputs)