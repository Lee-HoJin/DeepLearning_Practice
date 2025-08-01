import torch, numpy as np, pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import ( AutoTokenizer,
                           AutoModelForSequenceClassification,
                           TrainingArguments,
                           Trainer)
from sklearn.metrics import (f1_score, precision_score, recall_score)
from torch import cuda

# GPU 설정 - CPU 강제 설정 제거
device = 'cuda' if cuda.is_available() else 'cpu'
print(f"**Using {device} device**")

# GPU 메모리 최적화
if device == 'cuda':
    torch.cuda.empty_cache()

path="owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH"
dataset = load_dataset(path, split = "train[:10%]")
train_dataset = pd.DataFrame(dataset)

text_column = 'abstractText' # text field
label_names = list(train_dataset.columns[6:])
num_labels = len(label_names)
print('Number of Labels: ', num_labels)

print(train_dataset[[text_column] + label_names])

train_dataset["labels"]=train_dataset.apply(
    lambda x: x[label_names].to_numpy(), axis=1)
print(train_dataset[[text_column, "labels"]])

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 토큰화 함수 개선 - 최대 길이 제한 추가
def tokenize(batch):
    return tokenizer(batch[text_column],
                     padding=True,
                     truncation=True,
                     max_length=512)  # 최대 길이 명시

q = train_dataset[[text_column, "labels"]].copy()
CUT = int((q.shape)[0] * 0.5)
CUT2 = int((q.shape)[0] * 0.75)

train_df = q[:CUT] # training set
val_df = q[CUT : CUT2] # validation set
test_df = q[CUT2: ] # test set

train = Dataset.from_pandas(train_df) # cast to Dataset object
val = Dataset.from_pandas(val_df)
test = Dataset.from_pandas(test_df)

# 토큰화 - 배치 처리 최적화 (최신 버전 개선)
train_encoded = train.map(tokenize,
                          batched=True,
                          batch_size=1000,  # 배치 크기 명시
                          num_proc=4 if device == 'cuda' else 2,  # 멀티프로세싱
                          remove_columns=[col for col in train.column_names if col not in ['labels']])

val_encoded = val.map(tokenize,
                      batched=True,
                      batch_size=1000,
                      num_proc=4 if device == 'cuda' else 2,
                      remove_columns=[col for col in val.column_names if col not in ['labels']])

test_encoded = test.map(tokenize,
                        batched=True,
                        batch_size=1000,
                        num_proc=4 if device == 'cuda' else 2,
                        remove_columns=[col for col in test.column_names if col not in ['labels']])

def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    y_pred = y_pred.sigmoid() > 0.5
    y_true = y_true.bool()

    r = recall_score(y_true, y_pred, average='micro', pos_label=1, zero_division=0)
    p = precision_score(y_true, y_pred, average='micro', pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', pos_label=1, zero_division=0)
    result = {"Recall": r,
              "Precision": p,
              "F1": f1}
    
    return result

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss()
        
        # 라벨을 올바른 디바이스로 이동
        labels = labels.to(logits.device)
        
        preds_ = logits.view(-1, self.model.config.num_labels)
        labels_ = labels.float().view(-1, self.model.config.num_labels)
        
        loss = loss_fct(preds_, labels_)
        
        return (loss, outputs) if return_outputs else loss

# 배치 크기 조정 - GPU 메모리에 맞게
batch_size = 8 if device == 'cuda' else 4  # GPU에서 더 큰 배치 사용
num_epochs = 3

args = TrainingArguments(
    output_dir="/tmp",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    save_steps=100,
    eval_steps=100,
    save_strategy="steps",
    eval_strategy="steps",  # evaluation_strategy -> eval_strategy (최신 버전)
    # GPU 최적화 설정
    fp16=True if device == 'cuda' else False,  # 혼합 정밀도 사용
    dataloader_pin_memory=False,  # 메모리 핀닝 비활성화
    gradient_accumulation_steps=2,  # 그래디언트 누적으로 effective batch size 증가
    warmup_steps=100,
    logging_steps=50,
    remove_unused_columns=False,  # 라벨 컬럼 유지
    report_to="none"  # wandb, tensorboard 등 로깅 비활성화
)

# 모델 로드 및 GPU 이동
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=num_labels,
    problem_type="multi_label_classification"  # 멀티라벨 분류 명시
)

# 핵심: 모델을 GPU로 이동
model = model.to(device)

multi_trainer = MultilabelTrainer(
    model, args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("Starting training...")
multi_trainer.train()

print("Running predictions...")
res = multi_trainer.predict(test_encoded)
result_df = pd.Series(compute_metrics(res[:2])).to_frame()
print(result_df)

# GPU 메모리 정리 및 모니터링
if device == 'cuda':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability()}")