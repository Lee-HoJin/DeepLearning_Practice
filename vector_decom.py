import numpy as np
import matplotlib.pyplot as plt

class VectorDecomposition:
    def __init__(self):
        print("🔍 Vector Direction & Magnitude Decomposition")
        print("=" * 50)
    
    def decompose_vector(self, vector, name="Vector"):
        """벡터를 방향과 크기로 분해"""
        print(f"\n📊 {name} Analysis:")
        print(f"Original vector: {vector[:5]}... (showing first 5 dims)")
        
        # 크기 계산 (L2 norm)
        magnitude = np.linalg.norm(vector)
        print(f"Magnitude (||v||): {magnitude:.6f}")
        
        # 방향 계산 (단위벡터)
        if magnitude > 0:
            direction = vector / magnitude
            print(f"Direction vector norm: {np.linalg.norm(direction):.6f}")
            print(f"Direction vector: {direction[:5]}... (showing first 5 dims)")
        else:
            direction = np.zeros_like(vector)
            print("Zero vector - no direction")
        
        return magnitude, direction
    
    def compare_vectors(self):
        """여러 벡터들의 방향과 크기 비교"""
        print(f"\n🔄 Comparing Multiple Vectors")
        print("-" * 30)
        
        # 서로 다른 벡터들 생성
        np.random.seed(42)
        
        # 벡터 1: 일반적인 임베딩
        vec1 = np.random.normal(0, 0.1, 10)  # 10차원으로 단순화
        
        # 벡터 2: vec1의 2배 (같은 방향, 다른 크기)
        vec2 = 2 * vec1
        
        # 벡터 3: 다른 방향
        vec3 = np.random.normal(0, 0.1, 10)
        
        vectors = [vec1, vec2, vec3]
        names = ["Vec1", "Vec2 (2×Vec1)", "Vec3"]
        
        magnitudes = []
        directions = []
        
        for vec, name in zip(vectors, names):
            mag, direction = self.decompose_vector(vec, name)
            magnitudes.append(mag)
            directions.append(direction)
        
        # 방향 유사도 계산 (코사인 유사도)
        print(f"\n📐 Direction Similarities (Cosine):")
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                cosine_sim = np.dot(directions[i], directions[j])
                print(f"{names[i]} vs {names[j]}: {cosine_sim:.6f}")
        
        # 크기 비교
        print(f"\n📏 Magnitude Ratios:")
        for i in range(len(magnitudes)):
            for j in range(i+1, len(magnitudes)):
                ratio = magnitudes[i] / magnitudes[j]
                print(f"{names[i]}/{names[j]}: {ratio:.6f}")
        
        return vectors, magnitudes, directions
    
    def similarity_comparison(self):
        """코사인 vs 유클리드 거리 비교"""
        print(f"\n⚖️ Cosine vs Euclidean Distance Analysis")
        print("-" * 40)
        
        # 예시 벡터들
        base_vec = np.array([1, 2, 3, 4, 5])
        
        vectors = {
            "Original": base_vec,
            "2× Scaled": 2 * base_vec,
            "Different Direction": np.array([1, -2, 3, -4, 5]),
            "Similar Direction": np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        }
        
        print("Comparing all pairs:")
        names = list(vectors.keys())
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names[i+1:], i+1):
                vec1, vec2 = vectors[name1], vectors[name2]
                
                # 코사인 유사도
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                # 유클리드 거리
                eucl_dist = np.linalg.norm(vec1 - vec2)
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Cosine similarity: {cos_sim:.4f}")
                print(f"  Euclidean distance: {eucl_dist:.4f}")
    
    def word_embedding_example(self):
        """실제 단어 임베딩 상황 시뮬레이션"""
        print(f"\n📝 Word Embedding Example")
        print("-" * 30)
        
        # 가짜 단어 임베딩들 (512차원)
        np.random.seed(123)
        
        embeddings = {
            "king": np.random.normal(0.5, 0.1, 512),
            "queen": np.random.normal(0.4, 0.1, 512), 
            "man": np.random.normal(0.3, 0.1, 512),
            "woman": np.random.normal(0.25, 0.1, 512),
            "apple": np.random.normal(-0.2, 0.1, 512)  # 다른 의미 영역
        }
        
        # 각 단어의 방향과 크기 분석
        word_info = {}
        for word, emb in embeddings.items():
            magnitude = np.linalg.norm(emb)
            direction = emb / magnitude
            word_info[word] = {"magnitude": magnitude, "direction": direction, "embedding": emb}
            
            print(f"{word:6s}: magnitude = {magnitude:.4f}")
        
        # 단어 간 유사도 분석
        print(f"\n🔍 Word Similarity Analysis:")
        words = list(embeddings.keys())
        
        print("Cosine Similarities:")
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                emb1, emb2 = embeddings[word1], embeddings[word2]
                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                print(f"  {word1:6s} - {word2:6s}: {cos_sim:.4f}")
    
    def normalization_effects(self):
        """정규화의 효과 보기"""
        print(f"\n🎯 Normalization Effects")
        print("-" * 25)
        
        # 원본 벡터
        original = np.array([3, 4, 0, 0, 0])  # 크기 = 5
        
        # L2 정규화
        normalized = original / np.linalg.norm(original)
        
        print(f"Original vector: {original}")
        print(f"Original magnitude: {np.linalg.norm(original)}")
        print(f"\nNormalized vector: {normalized}")
        print(f"Normalized magnitude: {np.linalg.norm(normalized)}")
        
        # 다른 크기의 같은 방향 벡터
        scaled = 10 * original
        scaled_normalized = scaled / np.linalg.norm(scaled)
        
        print(f"\nScaled vector (10×): {scaled}")
        print(f"Scaled normalized: {scaled_normalized}")
        print(f"Same direction? {np.allclose(normalized, scaled_normalized)}")
    
    def run_all_examples(self):
        """모든 예시 실행"""
        self.compare_vectors()
        self.similarity_comparison()
        self.word_embedding_example()
        self.normalization_effects()

# 실행
if __name__ == "__main__":
    demo = VectorDecomposition()
    demo.run_all_examples()