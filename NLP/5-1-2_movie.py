from sys import stdin
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('movies_metadata.csv', low_memory=False)
# print(data.head(2))

## 🟢 'release_date' 열을 datetime 형식으로 변환
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')  # 오류 발생 시 NaT로 처리

## 🟢 1970년 이후 영화만 남기기
data = data[data['release_date'] >= '1970-01-01'].copy()


## preprocessing
data = data.head(20000)
data['overview'] = data['overview'].fillna("")
data['title'] = data['title'].fillna("")

## TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print('TF-IDF 행렬의 크기(shape) :',tfidf_matrix.shape)

# NumPy 배열을 PyTorch Tensor로 변환 (GPU로 이동)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfidf_tensor = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32).to(device)

## Dict 만들기
title_to_index = dict(zip(data['title'], data.index))

## 영화의 제목을 입력하면 코사인 유사도를 통해
## overview가 가장 유사한 10개의 영화를 찾아내는 함수
def get_recommendations(title) : 

    print("_선택한 영화 제목: \n", title)

    if title not in title_to_index :
        return "리스트에 없는 영화 제목입니다."

    # 선택한 영화의 제목으로부터 해당 영화의 인덱스를 받아옴
    idx = title_to_index[title]

    # 대상 영화 벡터 (1개 샘플)
    target_vec = tfidf_tensor[idx].unsqueeze(0)  # (1, N)

    # 코사인 유사도 계산 (전체 행렬과 비교)
    sim_scores = F.cosine_similarity(target_vec, tfidf_tensor)  # (N,)

    # 유사도가 높은 순서대로 정렬
    sim_scores = sim_scores.cpu().numpy()  # GPU -> CPU 변환 후 NumPy 배열
    top_indices = sim_scores.argsort()[::-1][1:11]  # 자기 자신 제외하고 10개 선택

    # 제목을 출력
    return data['title'].iloc[top_indices]

# print(get_recommendations('The Dark Knight Rises'))
# print(get_recommendations('God Father'))

while True :
    print("\n영화 제목을 입력해주세요: ")
    input = stdin.readline().rstrip()
    if input == '0' :
        break
    print(get_recommendations(input))