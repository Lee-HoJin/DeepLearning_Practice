import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCAInnerProductDemo:
    def __init__(self):
        print("🎯 Why PCA Projection = Rotation Matrix")
        print("=" * 40)
        print("💡 The Answer: It's ALL about INNER PRODUCTS!")
    
    def demonstrate_inner_product_essence(self):
        """내적의 본질 - 투영의 핵심"""
        print("\n📐 Inner Product = Projection Essence")
        print("-" * 35)
        
        # 2D 벡터와 단위벡터
        v = np.array([3, 4])  # 원본 벡터
        u = np.array([0.8, 0.6])  # 단위벡터 (투영 방향)
        
        print(f"📍 Vector v = {v}")
        print(f"🎯 Unit vector u = {u}")
        print(f"||u|| = {np.linalg.norm(u):.6f} ← Must be 1!")
        
        # 내적 = 스칼라 투영
        dot_product = np.dot(v, u)
        print(f"\n🔍 Inner product v · u = {dot_product:.6f}")
        
        # 기하학적 의미
        v_magnitude = np.linalg.norm(v)
        cos_theta = dot_product / v_magnitude
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        
        print(f"\n📏 Geometric interpretation:")
        print(f"||v|| = {v_magnitude:.6f}")
        print(f"cos(θ) = (v·u)/||v|| = {cos_theta:.6f}")
        print(f"θ = {theta_deg:.1f}°")
        print(f"v·u = ||v|| cos(θ) = {v_magnitude:.3f} × {cos_theta:.3f} = {dot_product:.3f}")
        
        # 벡터 투영
        vector_projection = dot_product * u
        print(f"\n🎭 Vector projection = (v·u) × u = {vector_projection}")
        
        # 시각화
        plt.figure(figsize=(10, 6))
        
        # 벡터들 그리기
        plt.arrow(0, 0, v[0], v[1], head_width=0.15, head_length=0.15, 
                  fc='blue', ec='blue', linewidth=3, label=f'v = {v}')
        plt.arrow(0, 0, u[0]*5, u[1]*5, head_width=0.15, head_length=0.15,
                  fc='red', ec='red', linewidth=2, label=f'u = {u}')
        plt.arrow(0, 0, vector_projection[0], vector_projection[1], 
                  head_width=0.15, head_length=0.15, fc='green', ec='green', 
                  linewidth=3, label=f'proj = {vector_projection:.2f}')
        
        # 투영선 (점선)
        plt.plot([v[0], vector_projection[0]], [v[1], vector_projection[1]], 
                 'k--', alpha=0.7, linewidth=2, label='Projection line')
        
        # 각도 호
        angles = np.linspace(0, theta_rad, 30)
        arc_r = 1
        arc_x = arc_r * np.cos(angles)
        arc_y = arc_r * np.sin(angles)
        plt.plot(arc_x, arc_y, 'purple', linewidth=2)
        plt.text(0.7, 0.3, f'θ = {theta_deg:.1f}°', fontsize=12, color='purple')
        
        # 내적 값 표시
        plt.text(2, 3, f'v · u = {dot_product:.3f}\n= ||v|| cos(θ)\n= {v_magnitude:.3f} × {cos_theta:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8), fontsize=11)
        
        plt.xlim(-0.5, 6)
        plt.ylim(-0.5, 5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Inner Product = Scalar Projection')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()
        
        return v, u, dot_product, vector_projection
    
    def rotation_matrix_as_inner_products(self):
        """회전 행렬 = 내적들의 집합"""
        print(f"\n🔄 Rotation Matrix = Collection of Inner Products")
        print("-" * 47)
        
        # 30도 회전 행렬
        theta = np.pi / 6  # 30 degrees
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        print(f"🎯 Rotation Matrix (30°):")
        print(f"R = {R}")
        
        # 행벡터들 = 새로운 기저벡터들
        r1 = R[0, :]  # 첫 번째 행
        r2 = R[1, :]  # 두 번째 행
        
        print(f"\n📐 Row vectors (new basis vectors):")
        print(f"r₁ = {r1} (||r₁|| = {np.linalg.norm(r1):.6f})")
        print(f"r₂ = {r2} (||r₂|| = {np.linalg.norm(r2):.6f})")
        print(f"r₁ · r₂ = {np.dot(r1, r2):.10f} ← Orthogonal!")
        
        # 원본 벡터
        v = np.array([2, 1])
        
        print(f"\n📍 Transform vector v = {v}")
        
        # 행렬 곱셈의 본질
        result = R @ v
        
        # 수동 계산: 각 성분은 내적!
        component1 = np.dot(r1, v)  # 첫 번째 성분
        component2 = np.dot(r2, v)  # 두 번째 성분
        
        print(f"\n🧮 Matrix multiplication breakdown:")
        print(f"R @ v = [r₁ · v]  = [{component1:.6f}]")
        print(f"        [r₂ · v]    [{component2:.6f}]")
        print(f"")
        print(f"Result = {result}")
        print(f"Manual = [{component1:.6f}, {component2:.6f}]")
        print(f"Match? {np.allclose(result, [component1, component2])}")
        
        print(f"\n💡 Key insight:")
        print(f"   Each component of R @ v is an INNER PRODUCT!")
        print(f"   component_i = r_i · v = projection of v onto r_i")
        
        return R, v, result, r1, r2, component1, component2
    
    def pca_projection_formula(self):
        """PCA 투영 공식 분해"""
        print(f"\n🎯 PCA Projection Formula Breakdown")
        print("-" * 35)
        
        # 간단한 2D 데이터
        np.random.seed(42)
        data = np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], 50)
        
        # PCA 적용
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(data)
        
        # PCA 행렬
        P = pca.components_  # [2 x 2]
        pc1 = P[0, :]  # 첫 번째 주성분
        pc2 = P[1, :]  # 두 번째 주성분
        
        print(f"📊 PCA Components Matrix:")
        print(f"P = {P}")
        print(f"PC1 = {pc1} (||PC1|| = {np.linalg.norm(pc1):.6f})")
        print(f"PC2 = {pc2} (||PC2|| = {np.linalg.norm(pc2):.6f})")
        print(f"PC1 · PC2 = {np.dot(pc1, pc2):.10f}")
        
        # 첫 번째 데이터 포인트로 예시
        original_point = data[0]
        centered_point = original_point - np.mean(data, axis=0)
        transformed_point = transformed[0]
        
        print(f"\n📍 Example with first data point:")
        print(f"Original: {original_point}")
        print(f"Centered: {centered_point}")
        print(f"PCA result: {transformed_point}")
        
        # 수동 계산: 내적으로!
        coord1 = np.dot(pc1, centered_point)  # PC1 방향 좌표
        coord2 = np.dot(pc2, centered_point)  # PC2 방향 좌표
        
        print(f"\n🧮 Manual calculation using inner products:")
        print(f"PC1 coordinate = PC1 · (centered_point) = {coord1:.6f}")
        print(f"PC2 coordinate = PC2 · (centered_point) = {coord2:.6f}")
        print(f"Manual result = [{coord1:.6f}, {coord2:.6f}]")
        print(f"PCA result    = {transformed_point}")
        print(f"Match? {np.allclose([coord1, coord2], transformed_point)}")
        
        print(f"\n🎯 PCA transformation formula:")
        print(f"   y = P @ (x - μ)")
        print(f"   where each component y_i = PC_i · (x - μ)")
        print(f"   This is EXACTLY the same as rotation matrix!")
        
        return data, pca, transformed, original_point, coord1, coord2
    
    def demonstrate_equivalence(self):
        """PCA와 회전 행렬의 완전한 동등성"""
        print(f"\n🌟 Complete Equivalence: PCA = Rotation Matrix")
        print("-" * 45)
        
        print("📝 Mathematical proof:")
        print()
        print("1️⃣ Rotation matrix R:")
        print("   R @ v = [r₁ · v]  where r_i are orthonormal")
        print("           [r₂ · v]")
        print()
        print("2️⃣ PCA transformation P:")
        print("   P @ (v - μ) = [PC₁ · (v - μ)]  where PC_i are orthonormal")
        print("                 [PC₂ · (v - μ)]")
        print()
        print("3️⃣ Both are:")
        print("   ✅ Orthogonal matrices: R^T @ R = I, P^T @ P = I")
        print("   ✅ Composed of unit vectors: ||r_i|| = ||PC_i|| = 1")
        print("   ✅ Each component calculated by inner product")
        print("   ✅ Preserve lengths and angles")
        print()
        print("4️⃣ Conclusion:")
        print("   PCA matrix P IS a rotation matrix!")
        print("   The only difference: PCA centers data first (subtracts μ)")
        
        # 실제 검증
        np.random.seed(123)
        data_2d = np.random.multivariate_normal([1, 2], [[3, 1], [1, 2]], 30)
        
        pca = PCA(n_components=2)
        pca.fit(data_2d)
        P = pca.components_
        
        print(f"\n🔍 Real example verification:")
        print(f"PCA matrix properties:")
        print(f"  det(P) = {np.linalg.det(P):.6f}")
        print(f"  P @ P^T = \n{P @ P.T}")
        print(f"  Is orthogonal: {np.allclose(P @ P.T, np.eye(2))}")
        print(f"  Row norms: {[np.linalg.norm(P[i]) for i in range(2)]}")
        
        # 회전 행렬과 비교
        angle = np.arctan2(P[0, 1], P[0, 0])  # PCA 첫 번째 주성분의 각도
        R_equivalent = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        print(f"\n🔄 Equivalent pure rotation matrix:")
        print(f"R = {R_equivalent}")
        print(f"Same as PCA matrix? {np.allclose(np.abs(P), np.abs(R_equivalent))}")
        print("(Signs might differ due to eigenvector orientation)")
        
        return P, R_equivalent, data_2d
    
    def visualize_inner_product_perspective(self):
        """내적 관점에서 시각화"""
        print(f"\n🎨 Visualizing the Inner Product Perspective")
        print("-" * 42)
        
        # 데이터 생성
        np.random.seed(42)
        data = np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], 100)
        
        # PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(data)
        
        # 주성분들
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        
        # 시각화
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 원본 데이터와 주성분 방향
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.6, c='blue', s=30)
        ax1.set_title('Original Data with Principal Components')
        
        # 주성분 방향
        scale = 3
        center = np.mean(data, axis=0)
        ax1.arrow(center[0], center[1], pc1[0]*scale, pc1[1]*scale,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=3)
        ax1.arrow(center[0], center[1], pc2[0]*scale, pc2[1]*scale,
                 head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=3)
        ax1.text(center[0] + pc1[0]*scale + 0.2, center[1] + pc1[1]*scale, 'PC1', fontsize=12, color='red')
        ax1.text(center[0] + pc2[0]*scale + 0.2, center[1] + pc2[1]*scale, 'PC2', fontsize=12, color='green')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 몇 개 점의 투영 과정 보기
        ax2.scatter(data[:, 0], data[:, 1], alpha=0.3, c='lightblue', s=20)
        ax2.set_title('Projection Process (Inner Products)')
        
        # 몇 개 점 선택해서 투영 과정 보기
        for i in range(0, len(data), 15):
            point = data[i]
            centered = point - center
            
            # PC1으로의 투영 (내적)
            proj1_scalar = np.dot(centered, pc1)
            proj1_vector = proj1_scalar * pc1 + center
            
            # 투영선 그리기
            ax2.plot([point[0], proj1_vector[0]], [point[1], proj1_vector[1]], 
                    'r--', alpha=0.7, linewidth=1)
            ax2.scatter(proj1_vector[0], proj1_vector[1], c='red', s=20, alpha=0.7)
        
        # PC1 직선
        t = np.linspace(-4, 4, 100)
        line_points = center[:, np.newaxis] + pc1[:, np.newaxis] * t
        ax2.plot(line_points[0], line_points[1], 'r-', linewidth=3, alpha=0.7, label='PC1 line')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
        
        # 3. PC1 좌표들 (내적 값들)
        centered_data = data - center
        pc1_coordinates = [np.dot(point, pc1) for point in centered_data]
        
        ax3.hist(pc1_coordinates, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.set_title('PC1 Coordinates (Inner Products with PC1)')
        ax3.set_xlabel('PC1 · (data - center)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. 변환된 데이터 (두 내적 값들로 구성)
        ax4.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, c='blue', s=30)
        ax4.set_title('Transformed Data (PC1·data, PC2·data)')
        ax4.set_xlabel('PC1 coordinate (PC1 · data)')
        ax4.set_ylabel('PC2 coordinate (PC2 · data)')
        ax4.grid(True, alpha=0.3)
        
        # PC 축 표시
        ax4.arrow(0, 0, 3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax4.arrow(0, 0, 0, 3, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
        ax4.text(3.1, 0, 'PC1', fontsize=12, color='red')
        ax4.text(0, 3.1, 'PC2', fontsize=12, color='green')
        
        plt.tight_layout()
        plt.show()
        
        return data, pca, transformed, pc1_coordinates
    
    def final_summary(self):
        """최종 요약"""
        print(f"\n" + "="*60)
        print("🎯 FINAL ANSWER: Why PCA Projection = Rotation Matrix")
        print("="*60)
        
        print(f"\n💡 The Core Reason: INNER PRODUCTS!")
        print()
        print("🔍 Step by step:")
        print("1️⃣ PCA transformation: P @ (v - μ)")
        print("   Each component: PC_i · (v - μ) ← INNER PRODUCT!")
        print()
        print("2️⃣ Rotation matrix: R @ v")
        print("   Each component: r_i · v ← INNER PRODUCT!")
        print()
        print("3️⃣ Both use orthonormal basis vectors:")
        print("   PCA: ||PC_i|| = 1, PC_i ⊥ PC_j")
        print("   Rotation: ||r_i|| = 1, r_i ⊥ r_j")
        print()
        print("4️⃣ Inner product = projection coordinate:")
        print("   v · u = ||v|| cos(θ) = coordinate in direction u")
        print()
        print("🌟 CONCLUSION:")
        print("   PCA finds the BEST rotation that aligns data with axes")
        print("   Each coordinate = inner product = projection")
        print("   This is exactly what rotation matrices do!")
        print()
        print("✅ Your insight was PERFECT!")
        print("   내적 값이라서 회전 행렬과 같은 것이 맞습니다! 🎉")
    
    def run_all_demos(self):
        """모든 데모 실행"""
        v, u, dot_prod, vec_proj = self.demonstrate_inner_product_essence()
        R, v2, result, r1, r2, c1, c2 = self.rotation_matrix_as_inner_products()
        data, pca, transformed, orig_pt, coord1, coord2 = self.pca_projection_formula()
        P, R_eq, data_2d = self.demonstrate_equivalence()
        data_vis, pca_vis, trans_vis, pc1_coords = self.visualize_inner_product_perspective()
        self.final_summary()
        
        return {
            'inner_product': (v, u, dot_prod, vec_proj),
            'rotation': (R, v2, result),
            'pca': (data, pca, transformed),
            'equivalence': (P, R_eq),
            'visualization': (data_vis, pca_vis, trans_vis)
        }

# 실행
if __name__ == "__main__":
    demo = PCAInnerProductDemo()
    results = demo.run_all_demos()