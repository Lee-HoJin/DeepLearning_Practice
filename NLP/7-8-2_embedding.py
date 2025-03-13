### 작동 안 될 듯

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 토큰화
tokenized_text = [['Hope', 'to', 'see', 'you', 'soon'], ['Nice', 'to', 'see', 'you', 'again']]

# 2. 각 단어에 대한 정수 인코딩
encoded_text = [[0, 1, 2, 3, 4],[5, 1, 2, 3, 6]]

# 3. 위 정수 인코딩 데이터가 아래의 임베딩 층의 입력이 된다.
vocab_size = 7
embedding_dim = 2

model = Sequential()
model.add(Embedding(vocab_size, output_dim, input_length))
