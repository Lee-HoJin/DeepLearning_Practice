import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

df = pd.read_csv('dart.csv',  sep=',')
df = df.dropna()

mecab = Mecab()

tagged_corpus_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):
  text = row['business']
  tag = row['name']
  tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))

print('문서의 수 :', len(tagged_corpus_list))

from gensim.models import doc2vec

model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)

# Vocabulary 빌드
model.build_vocab(tagged_corpus_list)

# Doc2Vec 학습
model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=20)

# 모델 저장
model.save('dart.doc2vec')
