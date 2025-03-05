from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


## Keras 전처리

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

sentences = sent_tokenize(raw_text)
# print(sentences)

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    preprocessed_sentences.append(result) 
print(preprocessed_sentences)

print('\n단어 집합:',vocab)

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences) 
print("\n인덱스 확인: ",tokenizer.word_index)
print("\n단어 개수:\n", tokenizer.word_counts)

print("\n매핑 결과\n", tokenizer.texts_to_sequences(preprocessed_sentences))

vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
# 패딩 때문에 숫자 0으로 지정된 단어가 없는데도 0부터 인덱싱
tokenizer.fit_on_texts(preprocessed_sentences)
# texts_to_sequences를 사용할 때 지정된 개수의 상위 단어만 남음

print("\n매핑 결과(상위 5개 단어만)\n", tokenizer.texts_to_sequences(preprocessed_sentences))

## OOV까지 추가

# 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)
print('\n단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))

print("\n매핑 결과(상위 5개 단어, OOV 추가)\n", tokenizer.texts_to_sequences(preprocessed_sentences))