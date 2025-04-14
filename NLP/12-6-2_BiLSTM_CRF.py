import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF  # pip install pytorch-crf
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
data = pd.read_csv("ner_dataset.csv", encoding="latin1")

print('Tag 열의 각각의 값의 개수 카운트')
print('================================')
print(data.groupby('Tag').size().reset_index(name='count'))

data = data.fillna(method="ffill")

data['Word'] = data['Word'].str.lower()
print(data[:5])

func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]
print("전체 샘플 개수: {}".format(len(tagged_sentences)))

print(tagged_sentences[0]) # 첫번째 샘플 출력

sentences, ner_tags = [], [] 
for tagged_sentence in tagged_sentences: # 47,959개의 문장 샘플을 1개씩 불러온다.

    # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentence, tag_info = zip(*tagged_sentence) 
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info)) # 각 샘플에서 개체명 태깅 정보만 저장한다.

def tokenize_texts(sentences):
    word_to_index = defaultdict(lambda: len(word_to_index))
    word_to_index["PAD"] = 0
    word_to_index["OOV"] = 1

    encoded = []
    for sentence in sentences:
        enc = [word_to_index[word] for word in sentence]
        encoded.append(enc)

    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return encoded, word_to_index, index_to_word

def tokenize_labels(tags):
    label_encoder = LabelEncoder()
    flat_tags = [tag for sent in tags for tag in sent]
    label_encoder.fit(flat_tags + ['PAD'])
    encoded = [[label_encoder.transform([tag])[0] for tag in sent] for sent in tags]
    index_to_tag = {i: tag for i, tag in enumerate(label_encoder.classes_)}
    return encoded, label_encoder, index_to_tag

X_data, word_to_index, index_to_word = tokenize_texts(sentences)
y_data, label_encoder, index_to_ner = tokenize_labels(ner_tags)

def pad_sequences_custom(sequences, max_len, padding_value=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [padding_value] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return np.array(padded)

max_len = 70
X_data = pad_sequences_custom(X_data, max_len)
y_data = pad_sequences_custom(y_data, max_len)

X_train, X_test, y_train_int, y_test_int = train_test_split(X_data, y_data, test_size=0.2, random_state=777)

class NERDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.LongTensor(inputs)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

batch_size = 32
vocab_size = len(word_to_index) 
tag_size = len(label_encoder.classes_)

train_dataset = NERDataset(X_train, y_train_int)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = NERDataset(X_test, y_test_int)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)  # 각 tag에 대한 score 계산
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x):  # 예측 (inference) 시 사용
        mask = x != 0  # 패딩 마스크
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions, mask=mask)  # 가장 확률 높은 y 시퀀스 반환

    def _get_emissions(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        return emissions  # shape: (batch_size, seq_len, tagset_size)

    def loss(self, x, tags):  # 학습 시 사용
        mask = x != 0
        emissions = self._get_emissions(x)
        return -self.crf(emissions, tags, mask=mask, reduction='mean')


model = BiLSTM_CRF(vocab_size=vocab_size, tagset_size=tag_size,
                   embedding_dim=100, hidden_dim=256)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    epoch_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = model.loss(inputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

def sequences_to_tags(sequences, index_to_ner):
    results = []
    for sequence in sequences:
        result = []
        for idx in sequence:
            tag = index_to_ner.get(idx, 'O')
            result.append(tag.replace('PAD', 'O'))
        results.append(result)
    return results

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        predictions = model(inputs)  # list of list (batch_size x seq_len)
        inputs = inputs.tolist()
        labels = labels.tolist()

        for pred_seq, true_seq, inp in zip(predictions, labels, inputs):
            temp_true, temp_pred = [], []
            for p, t, w in zip(pred_seq, true_seq, inp):
                if w != 0:  # PAD 제외
                    temp_true.append(index_to_ner[t].replace("PAD", "O"))
                    temp_pred.append(index_to_ner[p].replace("PAD", "O"))
            y_true.append(temp_true)
            y_pred.append(temp_pred)

# F1 및 리포트 출력
print("F1-score: {:.1%}".format(f1_score(y_true, y_pred)))
print(classification_report(y_true, y_pred))
