import torch, numpy as np, pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import ( AutoTokenizer,
                           AutoModelForSequenceClassification,
                           TrainingArguments,
                           Trainer)
from sklearn.metrics import (f1_score, precision_score, recall_score)


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
print(f"**Using {device} device**")

path="owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH"
dataset = load_dataset(path, split = "train[:10%]")
train_dataset = pd.DataFrame(dataset)

text_column = 'abstractText' # text field
label_names = list(train_dataset.columns[6:])
num_labels = len(label_names)
print('Number of Labels: ', num_labels)

print(train_dataset[[text_column] + label_names])

train_dataset["labels"]=train_dataset.apply( lambda x: x[label_names].to_numpy(), axis=1)
print(train_dataset[[text_column, "labels"]])

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize (batch) :
    return tokenizer(batch[text_column],
                     padding = True,
                     truncation = True)

q = train_dataset[[text_column, "labels"]].copy()
CUT = int((q.shape)[0] * 0.5)
CUT2 = int((q.shape)[0] * 0.75)

train_df = q[:CUT] # training set
val_df = q[CUT : CUT2] # validation set
test_df = q[CUT2: ] # test set

train = Dataset.from_pandas(train_df) # cast to Dataset object
val = Dataset.from_pandas(val_df)
test = Dataset.from_pandas(test_df)

train_encoded = train.map(tokenize,
                          batched = True,
                          batch_size = None)

val_encoded = val.map(tokenize,
                      batched = True,
                      batch_size = None)

test_encoded = test.map(tokenize,
                        batched = True,
                        batch_size = None)

def compute_metrics(eval_pred) :
    y_pred, y_true = eval_pred
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    y_pred = y_pred.sigmoid() > 0.5
    y_true = y_true.bool()

    r = recall_score(y_true, y_pred, average = 'micro', pos_label = 1)
    p = precision_score(y_true, y_pred, average = 'micro', pos_label = 1)
    f1 = f1_score(y_true, y_pred, average = 'micro', pos_label = 1)
    result = {"Recall": r,
              "Precision": p,
              "F1": f1}
    
    return result


class MultilabelTrainer(Trainer) :
    def compute_loss(self, model, inputs, return_outputs = False) :
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss()

        preds_ = logits.view(-1, self.model.config.num_labels)
        labels_ = labels.float().view(-1, self.model.config.num_labels)
        
        loss = loss_fct(preds_, labels_)

        return (loss, outputs) if return_outputs else loss

batch_size = 16
num_epochs = 3

args = TrainingArguments(
    output_dir="/tmp",
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_epochs,
    do_train = True,
    do_eval = True,
    load_best_model_at_end = True,
    save_steps = 100,
    eval_steps = 100,
    save_strategy = "steps",
    evaluation_strategy = "steps"
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = num_labels)

multi_trainer = MultilabelTrainer(
    model, args,
    train_dataset = train_encoded,
    eval_dataset = val_encoded,
    compute_metrics = compute_metrics,
    tokenizer = tokenizer
)

model = model.to(device)
multi_trainer.train()

res = multi_trainer.predict(test_encoded)
pd.Series(compute_metrics(res[:2])).to_frame()