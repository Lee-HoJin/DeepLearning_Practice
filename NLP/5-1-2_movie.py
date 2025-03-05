import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('movies_metadata.csv', low_memory=False)
# print(data.head(2))

## preprocessing
data['overview'] = data['overview'].fillna("")
data['title'] = data['title'].fillna("")

## TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print('TF-IDF 행렬의 크기(shape) :',tfidf_matrix.shape)

## 코사인 유사도
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('코사인 유사도 연산 결과(shape) :',cosine_sim.shape)

## Dict 만들기
title_to_index = dict(zip(data['title'], data.index))

## 영화의 제목을 입력하면 코사인 유사도를 통해
## overview가 가장 유사한 10개의 영화를 찾아내는 함수
def get_recommendations(title, cosine_sim = cosine_sim) : 

    print("선택한 영화 제목: \n", title)

    if title not in title_to_index :
        return "리스트에 없는 영화 제목입니다."

    # 선택한 영화의 제목으로부터 해당 영화의 인덱스를 받아옴
    idx = title_to_index[title]

    # 해당 영화와 모든 영화와의 유사도를 가져옴
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    # 가장 유사한 10개의 영화를 받아옴
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개 영화의 인덱스를 가져옴
    movie_indices = [idx[0] for idx in sim_scores]

    # 제목을 출력
    return data['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))