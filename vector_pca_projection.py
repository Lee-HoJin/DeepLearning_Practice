import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

class PCAProjectionExplained:
    def __init__(self):
        print("🎯 PCA Projection: Geometric & Mathematical Understanding")
        print("=" * 60)
    
    def demonstrate_1d_projection(self):
        """1차원 투영의 기본 개념"""
        print("\n📐 1D Projection Fundamentals")
        print("-" * 30)
        
        # 2D 점과 투영 방향
        point = np.array([3, 4])
        direction = np.array([1, 0])  # x축 방향 (단위벡터)
        
        print(f"📍 Point: {point}")
        print(f"🎯 Direction (unit vector): {direction}")
        print(f"Direction magnitude: {np.linalg.norm(direction):.6f}")
        
        # 스칼라 투영 (내적)
        scalar_projection = np.dot(point, direction)
        print(f"\n📏 Scalar projection = point · direction = {scalar_projection}")
        
        # 벡터 투영
        vector_projection = scalar_projection * direction
        print(f"🎭 Vector projection = {vector_projection}")
        
        # 기하학적 해석
        print(f"\n🎨 Geometric interpretation:")
        print(f"  - 점 {point}에서 방향 {direction}으로 수직선을 내림")
        print(f"  - 그 발(foot)이 {vector_projection}")
        print(f"  - 거리(좌표)가 {scalar_projection}")
        
        # 시각화
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(point[0], point[1], c='red', s=100, label='Original Point')
        plt.scatter(vector_projection[0], vector_projection[1], c='blue', s=100, label='Projection')
        
        # 투영선 그리기
        plt.plot([point[0], vector_projection[0]], [point[1], vector_projection[1]], 
                 'k--', alpha=0.5, label='Projection Line')
        
        # 방향 축 그리기
        plt.arrow(0, 0, direction[0]*5, direction[1]*5, head_width=0.1, 
                  head_length=0.1, fc='green', ec='green', label='Direction Axis')
        
        plt.xlim(-1, 6)
        plt.ylim(-1, 5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('1D Projection onto X-axis')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 1D 결과
        plt.subplot(1, 2, 2)
        plt.scatter(scalar_projection, 0, c='blue', s=100)
        plt.axhline(y=0, color='green', linewidth=3, alpha=0.7)
        plt.xlim(-1, 6)
        plt.ylim(-0.5, 0.5)
        plt.title('Projected Point on 1D Line')
        plt.xlabel('Projected Coordinate')
        plt.yticks([])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return point, direction, scalar_projection, vector_projection
    
    def pca_projection_math(self):
        """PCA 투영의 수학적 공식들"""
        print(f"\n🧮 PCA Projection Mathematics")
        print("-" * 32)
        
        print("📝 Step-by-step Mathematical Formulation:")
        print()
        print("1️⃣ Data Centering:")
        print("   X_centered = X - μ")
        print("   where μ = mean(X)")
        print()
        print("2️⃣ Principal Components (Eigenvectors):")
        print("   C = (1/n) X_centered^T X_centered  (covariance matrix)")
        print("   C v_i = λ_i v_i  (eigenvalue equation)")
        print("   where v_i = i-th principal component (unit vector)")
        print("         λ_i = i-th eigenvalue (variance explained)")
        print()
        print("3️⃣ Projection Formula:")
        print("   For a single point x_centered:")
        print("   projected_coordinate_i = x_centered · v_i")
        print("   projected_vector_i = (x_centered · v_i) × v_i")
        print()
        print("4️⃣ Matrix Form:")
        print("   Y = X_centered @ V^T")
        print("   where V = [v_1, v_2, ..., v_k] (k principal components)")
        print("         Y = projected coordinates matrix")
        print()
        print("5️⃣ Reconstruction:")
        print("   X_reconstructed = Y @ V + μ")
        
    def demonstrate_2d_to_1d_projection(self):
        """2D → 1D PCA 투영 실습"""
        print(f"\n🎯 2D → 1D PCA Projection Demo")
        print("-" * 33)
        
        # 타원형 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        # 상관관계가 있는 데이터
        mean = [2, 1]
        cov = [[2, 1.5], [1.5, 2]]
        data = np.random.multivariate_normal(mean, cov, n_samples)
        
        print(f"📊 Generated data shape: {data.shape}")
        print(f"Data mean: {np.mean(data, axis=0)}")
        
        # PCA 적용
        pca = PCA(n_components=1)
        projected_data = pca.fit_transform(data)
        
        # 수동 계산으로 검증
        center = np.mean(data, axis=0)
        centered_data = data - center
        pc1 = pca.components_[0]
        
        print(f"\n🎯 PCA Results:")
        print(f"Data center: {center}")
        print(f"1st Principal Component: {pc1}")
        print(f"PC1 magnitude: {np.linalg.norm(pc1):.6f}")
        print(f"Explained variance: {pca.explained_variance_[0]:.6f}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.3%}")
        
        # 수동 투영 계산
        manual_projections = []
        for i in range(min(5, len(data))):
            point = data[i]
            centered_point = point - center
            
            # 스칼라 투영
            scalar_proj = np.dot(centered_point, pc1)
            
            # 벡터 투영 (2D 공간에서)
            vector_proj = scalar_proj * pc1
            
            # 원래 좌표계로 복원
            reconstructed_point = vector_proj + center
            
            manual_projections.append(scalar_proj)
            
            print(f"\n📍 Point {i+1}:")
            print(f"  Original: [{point[0]:6.3f}, {point[1]:6.3f}]")
            print(f"  Centered: [{centered_point[0]:6.3f}, {centered_point[1]:6.3f}]")
            print(f"  Scalar projection: {scalar_proj:6.3f}")
            print(f"  PCA result: {projected_data[i, 0]:6.3f}")
            print(f"  Match? {np.isclose(scalar_proj, projected_data[i, 0])}")
        
        # 시각화
        plt.figure(figsize=(15, 5))
        
        # 원본 데이터와 투영
        plt.subplot(1, 3, 1)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, c='blue', s=30, label='Original Data')
        plt.scatter(center[0], center[1], c='red', s=100, marker='x', linewidth=3, label='Center')
        
        # PC1 방향 표시
        scale = 3
        plt.arrow(center[0], center[1], pc1[0]*scale, pc1[1]*scale,
                  head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2,
                  label='PC1 Direction')
        
        plt.title('Original 2D Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 투영 과정 시각화
        plt.subplot(1, 3, 2)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.3, c='lightblue', s=20)
        
        # 몇 개 점의 투영 과정 보기
        for i in range(0, len(data), 20):
            point = data[i]
            centered = point - center
            scalar_proj = np.dot(centered, pc1)
            vector_proj = scalar_proj * pc1 + center
            
            plt.plot([point[0], vector_proj[0]], [point[1], vector_proj[1]], 
                     'r--', alpha=0.5, linewidth=1)
            plt.scatter(vector_proj[0], vector_proj[1], c='red', s=30, alpha=0.7)
        
        # PC1 직선
        t = np.linspace(-3, 3, 100)
        line_points = center[:, np.newaxis] + pc1[:, np.newaxis] * t
        plt.plot(line_points[0], line_points[1], 'g-', linewidth=3, alpha=0.7, label='PC1 Line')
        
        plt.title('Projection Process')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 1D 결과
        plt.subplot(1, 3, 3)
        plt.hist(projected_data[:, 0], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.title('Projected 1D Data Distribution')
        plt.xlabel('Coordinate on PC1')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return data, pca, projected_data
    
    def demonstrate_3d_to_2d_projection(self):
        """3D → 2D PCA 투영"""
        print(f"\n🌟 3D → 2D PCA Projection")
        print("-" * 25)
        
        # 3D 데이터 생성 (평면에 가까운 분포)
        np.random.seed(123)
        n_samples = 200
        
        # 주로 XY 평면에 분포하는 데이터
        x = np.random.normal(0, 2, n_samples)
        y = np.random.normal(0, 2, n_samples)
        z = 0.3 * x + 0.2 * y + np.random.normal(0, 0.5, n_samples)  # 약간의 Z 성분
        
        data_3d = np.column_stack([x, y, z])
        
        print(f"📊 3D Data shape: {data_3d.shape}")
        print(f"Data mean: {np.mean(data_3d, axis=0)}")
        print(f"Data std: {np.std(data_3d, axis=0)}")
        
        # PCA 적용
        pca_3d = PCA(n_components=2)
        projected_2d = pca_3d.fit_transform(data_3d)
        
        print(f"\n🎯 PCA Results:")
        print(f"PC1: {pca_3d.components_[0]}")
        print(f"PC2: {pca_3d.components_[1]}")
        print(f"Explained variance ratio: {pca_3d.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca_3d.explained_variance_ratio_):.3%}")
        
        # 시각화
        fig = plt.figure(figsize=(15, 5))
        
        # 3D 원본 데이터
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], 
                   alpha=0.6, c=data_3d[:, 2], cmap='viridis', s=20)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original 3D Data')
        
        # 주성분 방향 표시
        center = np.mean(data_3d, axis=0)
        scale = 2
        for i, (pc, color) in enumerate(zip(pca_3d.components_, ['red', 'blue'])):
            ax1.quiver(center[0], center[1], center[2],
                      pc[0]*scale, pc[1]*scale, pc[2]*scale,
                      color=color, arrow_length_ratio=0.1, linewidth=3,
                      label=f'PC{i+1}')
        ax1.legend()
        
        # 2D 투영 결과
        ax2 = fig.add_subplot(132)
        scatter = ax2.scatter(projected_2d[:, 0], projected_2d[:, 1], 
                             alpha=0.6, c=data_3d[:, 2], cmap='viridis', s=20)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Projected 2D Data')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Original Z value')
        
        # 분산 설명 비율
        ax3 = fig.add_subplot(133)
        components = ['PC1', 'PC2', 'Lost']
        ratios = list(pca_3d.explained_variance_ratio_) + [1 - sum(pca_3d.explained_variance_ratio_)]
        colors = ['red', 'blue', 'gray']
        
        ax3.bar(components, ratios, color=colors, alpha=0.7)
        ax3.set_ylabel('Variance Ratio')
        ax3.set_title('Variance Explained')
        ax3.grid(True, alpha=0.3)
        
        # 비율 텍스트 추가
        for i, (comp, ratio) in enumerate(zip(components, ratios)):
            ax3.text(i, ratio + 0.01, f'{ratio:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return data_3d, pca_3d, projected_2d
    
    def reconstruction_demo(self):
        """투영된 데이터의 복원"""
        print(f"\n🔄 Reconstruction from Projection")
        print("-" * 32)
        
        # 2D 데이터
        np.random.seed(42)
        original_2d = np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], 50)
        
        # 1D로 투영
        pca = PCA(n_components=1)
        projected_1d = pca.fit_transform(original_2d)
        
        # 복원
        reconstructed_2d = pca.inverse_transform(projected_1d)
        
        # 수동 복원 확인
        center = np.mean(original_2d, axis=0)
        pc1 = pca.components_[0]
        
        manual_reconstructed = []
        for proj_coord in projected_1d[:, 0]:
            # 1D 좌표를 2D로 복원
            vector_in_2d = proj_coord * pc1  # PC1 방향으로 벡터 생성
            point_in_2d = vector_in_2d + center  # 중심점 더하기
            manual_reconstructed.append(point_in_2d)
        
        manual_reconstructed = np.array(manual_reconstructed)
        
        print(f"📊 Reconstruction Analysis:")
        print(f"Original shape: {original_2d.shape}")
        print(f"Projected shape: {projected_1d.shape}")
        print(f"Reconstructed shape: {reconstructed_2d.shape}")
        
        # 복원 오차 계산
        reconstruction_error = np.mean(np.linalg.norm(original_2d - reconstructed_2d, axis=1))
        print(f"Mean reconstruction error: {reconstruction_error:.6f}")
        
        # 수동 복원과 비교
        manual_match = np.allclose(reconstructed_2d, manual_reconstructed)
        print(f"Manual reconstruction matches sklearn: {manual_match}")
        
        # 시각화
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(original_2d[:, 0], original_2d[:, 1], c='blue', alpha=0.7, s=50, label='Original')
        plt.title('Original 2D Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(1, 3, 2)
        plt.scatter(projected_1d[:, 0], np.zeros_like(projected_1d[:, 0]), 
                   c='red', alpha=0.7, s=50)
        plt.axhline(y=0, color='black', linewidth=2)
        plt.title('Projected 1D Data')
        plt.xlabel('PC1 Coordinate')
        plt.ylabel('(Collapsed dimension)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(original_2d[:, 0], original_2d[:, 1], c='blue', alpha=0.5, s=50, label='Original')
        plt.scatter(reconstructed_2d[:, 0], reconstructed_2d[:, 1], c='red', alpha=0.7, s=30, label='Reconstructed')
        
        # 오차 벡터 표시
        for i in range(0, len(original_2d), 5):
            plt.plot([original_2d[i, 0], reconstructed_2d[i, 0]], 
                    [original_2d[i, 1], reconstructed_2d[i, 1]], 
                    'k--', alpha=0.5, linewidth=1)
        
        plt.title('Original vs Reconstructed')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        return original_2d, projected_1d, reconstructed_2d
    
    def run_all_demos(self):
        """모든 데모 실행"""
        # 기본 투영 개념
        point, direction, scalar_proj, vector_proj = self.demonstrate_1d_projection()
        
        # 수학적 공식
        self.pca_projection_math()
        
        # 2D → 1D 투영
        data_2d, pca_2d, proj_1d = self.demonstrate_2d_to_1d_projection()
        
        # 3D → 2D 투영  
        data_3d, pca_3d, proj_2d = self.demonstrate_3d_to_2d_projection()
        
        # 복원
        orig, proj, recon = self.reconstruction_demo()
        
        return {
            'basic': (point, direction, scalar_proj, vector_proj),
            '2d_to_1d': (data_2d, pca_2d, proj_1d),
            '3d_to_2d': (data_3d, pca_3d, proj_2d),
            'reconstruction': (orig, proj, recon)
        }

# 실행
if __name__ == "__main__":
    demo = PCAProjectionExplained()
    results = demo.run_all_demos()