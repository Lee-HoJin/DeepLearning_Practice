from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import pandas as pd
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader

"""
refactoring T5 to transformers library
"""


# 1. 모델과 토크나이저 로드
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 2. 데이터 준비 (기존 코드와 동일)
sst2_df = pd.DataFrame(load_dataset("glue", "sst2", split="train[:1000]"))
sst2_df = sst2_df[["sentence", "label"]]
sst2_df.columns = ["source_text", "target_text"]
prompt_sentiment = ". What is the sentiment ? " + "Good or Bad ?"
sst2_df["source_text"] = sst2_df.source_text.apply(lambda x: x + prompt_sentiment)
sst2_df["target_text"] = sst2_df.target_text.apply(lambda x: "Good" if x == 1 else "Bad")

opus = load_dataset("opus100", "de-en", split="train[:1000]")
opus_df = pd.DataFrame(opus)
opus_df["source_text"] = opus_df.apply(lambda x: x.translation["en"], axis=1)
opus_df["target_text"] = opus_df.apply(lambda x: x.translation["de"], axis=1)
opus_df = opus_df[["source_text", "target_text"]]
prompt_tr = ". Translate English to German"
opus_df["source_text"] = opus_df.source_text.apply(lambda x: x + prompt_tr)

merge = pd.concat([opus_df, sst2_df]).sample(frac=1.0).reset_index(drop=True)
print(merge.head(5))

train_df = merge[:1800]
eval_df = merge[1800:]

# 3. 데이터 전처리 함수
def preprocess_function(examples):
    inputs = examples["source_text"]
    targets = examples["target_text"]
    
    # 입력 토큰화
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding=True, 
        return_tensors="pt"
    )
    
    # 타겟 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
    
    # T5는 decoder_start_token_id를 자동으로 추가하므로 labels만 설정
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# 4. Dataset 클래스 생성
class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_len_input=512, max_len_target=128):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        source_text = str(self.data.iloc[index]['source_text'])
        target_text = str(self.data.iloc[index]['target_text'])
        
        # 입력 토큰화
        source = self.tokenizer.batch_encode_plus(
            [source_text], 
            max_length=self.max_len_input,
            pad_to_max_length=True,
            truncation=True, 
            padding="max_length",
            return_tensors='pt'
        )
        
        # 타겟 토큰화
        target = self.tokenizer.batch_encode_plus(
            [target_text], 
            max_length=self.max_len_target,
            pad_to_max_length=True, 
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        
        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': target_ids
        }

# 5. 데이터셋 생성
train_dataset = T5Dataset(train_df, tokenizer)
eval_dataset = T5Dataset(eval_df, tokenizer)

# 6. 훈련 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    warmup_steps=100,
    learning_rate=3e-4,
    fp16=False,  # precision=32와 동일
    dataloader_pin_memory=False,
)

# 7. Trainer 생성 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Starting training...")
trainer.train()

# 8. 추론 함수
def predict(text, model, tokenizer, device, max_length=128):
    model.eval()
    with torch.no_grad():
        # 입력 토큰화
        inputs = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        inputs = inputs.to(device)
        
        # 생성
        outputs = model.generate(
            inputs, 
            max_length=max_length,
            num_beams=2,
            early_stopping=True,
            temperature=0.7
        )
        
        # 디코딩
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction

# 9. 테스트
model.to('cpu')  # CPU로 이동 (기존 코드처럼)
a_sentence = "The cats are fun!"
result = predict(a_sentence + prompt_sentiment, model, tokenizer, torch.device('cpu'))
print(f"Input: {a_sentence + prompt_sentiment}")
print(f"Output: {result}")