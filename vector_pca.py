import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCAGeometricDemo:
    def __init__(self):
        print("🎯 PCA Geometric Understanding")
        print("=" * 40)
    
    def create_2d_data(self):
        """2D 데이터 생성 (타원형 분포)"""
        np.random.seed(42)
        n_samples = 200
        
        # 상관관계가 있는 데이터 생성
        mean = [0, 0]
        cov = [[3, 1.5],    # x 분산=3, 공분산=1.5
               [1.5, 1]]    # y 분산=1, 공분산=1.5
        
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return data
    
    def visualize_pca_process(self):
        """PCA 과정을 시각적으로 보여주기"""
        print("\n📊 PCA Geometric Interpretation")
        print("-" * 32)
        
        data = self.create_2d_data()
        
        # PCA 적용
        pca = PCA(n_components=2)
        pca.fit(data)
        
        # 주성분들
        pc1 = pca.components_[0]  # 첫 번째 주성분 (최대 분산 방향)
        pc2 = pca.components_[1]  # 두 번째 주성분 (PC1에 직교)
        
        print(f"Data shape: {data.shape}")
        print(f"Data center: [{np.mean(data[:,0]):.3f}, {np.mean(data[:,1]):.3f}]")
        
        print(f"\n🎯 Principal Components:")
        print(f"PC1 (1st component): [{pc1[0]:6.3f}, {pc1[1]:6.3f}]")
        print(f"PC1 magnitude: {np.linalg.norm(pc1):.6f} ← Unit vector!")
        print(f"PC2 (2nd component): [{pc2[0]:6.3f}, {pc2[1]:6.3f}]")
        print(f"PC2 magnitude: {np.linalg.norm(pc2):.6f} ← Unit vector!")
        
        # 직교성 확인
        orthogonal = np.dot(pc1, pc2)
        print(f"PC1 · PC2 (orthogonality): {orthogonal:.10f} ← Should be ~0")
        
        # 분산 정보
        print(f"\n📈 Variance Information:")
        print(f"Explained variance: {pca.explained_variance_}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.6f}")
        
        # 데이터 중심
        center = np.mean(data, axis=0)
        
        # 시각화
        plt.figure(figsize=(12, 5))
        
        # 원본 데이터와 주성분
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, c='blue', s=20)
        plt.scatter(center[0], center[1], c='red', s=100, marker='x', linewidth=3)
        
        # 주성분 방향 표시 (화살표)
        scale = 3  # 시각화를 위한 스케일링
        plt.arrow(center[0], center[1], pc1[0]*scale, pc1[1]*scale, 
                  head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2,
                  label=f'PC1 (explains {pca.explained_variance_ratio_[0]:.1%})')
        plt.arrow(center[0], center[1], pc2[0]*scale, pc2[1]*scale,
                  head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2,
                  label=f'PC2 (explains {pca.explained_variance_ratio_[1]:.1%})')
        
        plt.title('Original Data with Principal Components')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 1차원으로 투영 (PC1 방향으로만)
        plt.subplot(1, 2, 2)
        
        # PC1 방향으로 투영된 좌표들
        projected_1d = np.dot(data - center, pc1)  # 스칼라 값들
        
        # 1D 히스토그램으로 분포 보기
        plt.hist(projected_1d, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.title('Data Projected onto PC1 (2D → 1D)')
        plt.xlabel('Projected coordinate on PC1')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return data, pca, projected_1d
    
    def demonstrate_projection_math(self):
        """투영 수학 자세히 설명"""
        print(f"\n🧮 Projection Mathematics")
        print("-" * 25)
        
        data = self.create_2d_data()
        pca = PCA(n_components=1)  # 1차원으로 축소
        projected = pca.fit_transform(data)
        
        pc1 = pca.components_[0]  # 첫 번째 주성분
        center = np.mean(data, axis=0)
        
        print(f"Principal component (unit vector): {pc1}")
        print(f"Data center: {center}")
        
        # 몇 개 샘플의 투영 과정 보기
        print(f"\n📍 Manual Projection Examples:")
        for i in range(3):
            original_point = data[i]
            centered_point = original_point - center
            
            # 수동 투영: (v · u) where u is unit vector
            manual_projection = np.dot(centered_point, pc1)
            sklearn_projection = projected[i, 0]
            
            print(f"\nSample {i+1}:")
            print(f"  Original point: [{original_point[0]:6.3f}, {original_point[1]:6.3f}]")
            print(f"  Centered point: [{centered_point[0]:6.3f}, {centered_point[1]:6.3f}]")
            print(f"  Manual projection: {manual_projection:6.3f}")
            print(f"  Sklearn projection: {sklearn_projection:6.3f}")
            print(f"  Match? {np.isclose(manual_projection, sklearn_projection)}")
    
    def compare_different_directions(self):
        """다른 방향들과 분산 비교"""
        print(f"\n🎯 Why PC1 is Optimal Direction")
        print("-" * 30)
        
        data = self.create_2d_data()
        center = np.mean(data, axis=0)
        centered_data = data - center
        
        # PCA 결과
        pca = PCA(n_components=1)
        pca.fit(data)
        optimal_direction = pca.components_[0]
        
        # 여러 방향들 테스트
        angles = np.linspace(0, np.pi, 9)  # 0도부터 180도까지
        directions = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        
        variances = []
        for direction in directions:
            # 각 방향으로 투영
            projections = np.dot(centered_data, direction)
            variance = np.var(projections)
            variances.append(variance)
        
        print(f"Optimal direction (PC1): [{optimal_direction[0]:6.3f}, {optimal_direction[1]:6.3f}]")
        print(f"Optimal variance: {pca.explained_variance_[0]:.6f}")
        
        print(f"\nTesting different directions:")
        for i, (angle, direction, variance) in enumerate(zip(angles, directions, variances)):
            angle_deg = np.degrees(angle)
            print(f"  {angle_deg:3.0f}°: direction=[{direction[0]:6.3f}, {direction[1]:6.3f}], "
                  f"variance={variance:6.3f}")
        
        max_variance_idx = np.argmax(variances)
        print(f"\nMaximum variance achieved at {np.degrees(angles[max_variance_idx]):3.0f}°")
        print(f"This matches our PC1 direction: "
              f"{np.allclose(directions[max_variance_idx], optimal_direction, atol=0.1)}")
    
    def normalization_deep_dive(self):
        """정규화에 대한 깊이 있는 분석"""
        print(f"\n🔬 PCA Normalization Deep Dive")
        print("-" * 32)
        
        data = self.create_2d_data()
        pca = PCA(n_components=2)
        pca.fit(data)
        
        print("✅ PCA Components are Unit Vectors:")
        for i, component in enumerate(pca.components_):
            magnitude = np.linalg.norm(component)
            print(f"  PC{i+1}: {component} → ||PC{i+1}|| = {magnitude:.10f}")
        
        print(f"\n🔍 What about the magnitude information?")
        print(f"Explained variance (eigenvalues): {pca.explained_variance_}")
        print(f"Standard deviations: {np.sqrt(pca.explained_variance_)}")
        
        # 실제 고유벡터 vs PCA components
        # 공분산 행렬 직접 계산
        centered_data = data - np.mean(data, axis=0)
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 내림차순 정렬
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"\n🔬 Manual Eigendecomposition:")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Eigenvectors (columns):")
        for i in range(eigenvectors.shape[1]):
            eigenvec = eigenvectors[:, i]
            print(f"  Eigenvector {i+1}: {eigenvec}")
            print(f"  Magnitude: {np.linalg.norm(eigenvec):.10f}")
        
        print(f"\n🎯 Comparison:")
        print(f"PCA explained_variance vs Eigenvalues: "
              f"{np.allclose(pca.explained_variance_, eigenvalues)}")
        
        # 방향 비교 (부호는 다를 수 있음)
        for i in range(2):
            pca_comp = pca.components_[i]
            eigenvec = eigenvectors[:, i]
            
            # 부호 보정 (첫 번째 성분이 양수가 되도록)
            if pca_comp[0] < 0:
                pca_comp = -pca_comp
            if eigenvec[0] < 0:
                eigenvec = -eigenvec
                
            match = np.allclose(pca_comp, eigenvec)
            print(f"PC{i+1} vs Eigenvector{i+1}: {match}")
    
    def answer_user_questions(self):
        """사용자 질문에 대한 명확한 답변"""
        print(f"\n" + "="*50)
        print("🎯 ANSWERS TO YOUR QUESTIONS")
        print("="*50)
        
        print(f"\n1️⃣ 2D → 1D 축소의 geometric 의미:")
        print(f"   ✅ YES! 데이터 분포를 가장 잘 표현하는 하나의 방향(직선)을 찾는 것")
        print(f"   📍 이 방향으로 투영했을 때 분산이 최대가 되는 방향")
        print(f"   🎭 타원형 데이터에서 장축 방향을 찾는 것과 같음")
        
        print(f"\n2️⃣ PCA 주성분이 이미 normalized인가:")
        print(f"   ✅ YES! 모든 주성분(principal components)은 단위벡터")
        print(f"   📏 ||PC_i|| = 1.0 (정확히)")
        print(f"   🔍 방향만 나타내고, 크기 정보는 explained_variance_에 따로 저장")
        print(f"   💡 따라서 이미 정규화(normalized)되어 있다고 볼 수 있음")
    
    def run_all_demos(self):
        """모든 데모 실행"""
        data, pca, projected = self.visualize_pca_process()
        self.demonstrate_projection_math()
        self.compare_different_directions()
        self.normalization_deep_dive()
        self.answer_user_questions()
        
        return data, pca, projected

# 실행
if __name__ == "__main__":
    demo = PCAGeometricDemo()
    results = demo.run_all_demos()