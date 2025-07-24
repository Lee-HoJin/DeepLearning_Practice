import datasets
import numpy as np 
import json
from transformers import DistilBertTokenizerFast, \
                         AutoModelForTokenClassification, \
                         TrainingArguments, \
                         Trainer,  \
                         DataCollatorForTokenClassification, \
                         pipeline


conll2003 = datasets.load_dataset("conll2003")

# print(conll2003["train"][0])
# print(conll2003["train"].features["ner_tags"])

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


## 토큰과 레이블들을 align 해주는 함수
## 토큰이 pieces로 더 나뉘게 되는데 단어는 하나의 piece여야 하기 때문
label_all_tokens = True
def tokenize_and_align_labels(examples) :
    tokenized_inputs = tokenizer(examples["tokens"],
                                    truncation = True,
                                    is_split_into_words = True)
    
    labels = []

    for i, label in enumerate(examples["ner_tags"]) :
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids :
            if word_idx is None :
                label_ids.append(-100)

            elif word_idx != previous_word_idx :
                label_ids.append(label[word_idx])

            else :
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# q = tokenize_and_align_labels(conll2003['train'][4:5])

# for token, label in zip( tokenizer.convert_ids_to_tokens( q["input_ids"][0]), q["labels"][0]):
#     print(f"{token:_<40} {label}")

tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched = True)

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = 9)
args = TrainingArguments(
    "test-ner",
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_training_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 3,
    weight_decay = 0.01
)

data_collator =  DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

example = conll2003['train'][0]
label_list = conll2003["train"].features["ner_tags"].feature.names
labels = [label_list[i] for i in example["ner_tags"]]
metric.compute(predictions=[labels], references=[labels])


def compute_metrics(p): 
    predictions, labels = p 
    predictions = np.argmax(predictions, axis=2) 
    true_predictions = [ 
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(predictions, labels) 
    ] 
    true_labels = [ 
      [label_list[l] for (p, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(predictions, labels) 
   ] 
    results = metric.compute(predictions=true_predictions, references=true_labels) 
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  } 

trainer = Trainer( 
    model, 
    args, 
   train_dataset=tokenized_datasets["train"], 
   eval_dataset=tokenized_datasets["validation"], 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics 
) 

trainer.train() 

model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")

id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}


config = json.load(open("ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id

json.dump(config, open("ner_model/config.json","w"))

mmodel = AutoModelForTokenClassification.from_pretrained("ner_model")

nlp = pipeline("ner", model=mmodel, tokenizer=tokenizer)
example = "I live in Istanbul"

ner_results = nlp(example)
print(ner_results)