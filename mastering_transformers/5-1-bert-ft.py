from torch import cuda
from transformers import DistilBertTokenizerFast, \
                         DistilBertForSequenceClassification, \
                         TrainingArguments, \
                         Trainer, \
                         pipeline, \
                         DistilBertForSequenceClassification, \
                         DistilBertTokenizerFast \

from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


device = 'cuda' if cuda.is_available() else 'cpu'

model_path= 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})

imdb_train= load_dataset('imdb', split="train[:2000]+train[-2000:]")
imdb_test= load_dataset('imdb', split="test[:500]+test[-500:]")
imdb_val= load_dataset('imdb', split="test[500:1000]+test[-1000:-500]")

print(imdb_train.shape, imdb_test.shape, imdb_val.shape)

enc_train = imdb_train.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_test =  imdb_test.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_val =   imdb_val.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./MyIMDBModel', 
    do_train=True,
    do_eval=True,
    
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,
    
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps', 
    
    # TensorBoard log directory               
    logging_dir='./logs',            
    logging_steps=50,
    
    # other options : no, steps
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=cuda.is_available(),
    load_best_model_at_end=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,
    # training and validation dataset                 
    train_dataset=enc_train,         
    eval_dataset=enc_val,            
    compute_metrics= compute_metrics
)

results=trainer.train()

print(results)

q=[trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]

# saving the best fine-tuned model & tokenizer
model_save_path = "MyBestIMDBModel"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=250, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"].to(device),inputs["attention_mask"].to(device))
    probs = outputs[0].softmax(1)
    return probs, probs.argmax()

model.to(device)
text = "I didn't like the movie since it bored me "
print(get_prediction(text)[1].item())

model = DistilBertForSequenceClassification.from_pretrained("MyBestIMDBModel")
tokenizer= DistilBertTokenizerFast.from_pretrained("MyBestIMDBModel")
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print("the movie was very impressive")
print("sentiment prediction", nlp("the movie was very impressive"))

print("\nthe script of the picture was very poor")
print("sentiment prediction", nlp("the script of the picture was very poor"))