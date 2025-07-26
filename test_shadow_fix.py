# 测试阴影计算修正效果
import numpy as np
import matplotlib.pyplot as plt
from m11_optimized import OptimizedShadowCalculator, read_data
import pickle

def test_shadow_calculation():
    """测试修正后的阴影计算"""
    print("测试阴影计算修正效果...")
    
    # 使用前100个定日镜进行快速测试
    location = read_data()[:100]  # 只取前100个
    print(f"使用 {len(location)} 个定日镜进行测试")
    
    # 创建计算器
    calculator = OptimizedShadowCalculator(location, 6, 6, 4)
    
    # 模拟一个时间步的数据
    n_heliostats = len(location)
    
    # 简化的几何数据
    n_dingri = np.tile([0, 0, 1], (n_heliostats, 1))  # 法向量都向上
    v1 = np.tile([1, 0, 0], (n_heliostats, 1))
    v2 = np.tile([0, 1, 0], (n_heliostats, 1))
    
    # 计算角点
    corners = np.zeros((n_heliostats, 4, 3))
    W, H, h = 6, 6, 4
    
    corners[:, 0] = np.column_stack([
        location[:, 0] + W/2, location[:, 1] + H/2, np.full(n_heliostats, h)
    ])
    corners[:, 1] = np.column_stack([
        location[:, 0] - W/2, location[:, 1] + H/2, np.full(n_heliostats, h)
    ])
    corners[:, 2] = np.column_stack([
        location[:, 0] - W/2, location[:, 1] - H/2, np.full(n_heliostats, h)
    ])
    corners[:, 3] = np.column_stack([
        location[:, 0] + W/2, location[:, 1] - H/2, np.full(n_heliostats, h)
    ])
    
    time_data = {
        'n_dingri': n_dingri,
        'v1': v1,
        'v2': v2,
        'corners': corners,
        's_in_current': np.array([0, 0.7071, -0.7071])  # 45度太阳高度角
    }
    
    # 计算所有定日镜的阴影
    shadows = []
    for k in range(n_heliostats):
        shadow_ratio = calculator.calculate_single_heliostat_shadow(k, time_data)
        shadows.append(shadow_ratio)
        if k % 20 == 0:
            print(f"计算定日镜 {k}: 阴影比例 = {shadow_ratio:.4f}")
    
    shadows = np.array(shadows)
    
    print(f"\n阴影计算结果统计:")
    print(f"最小值: {np.min(shadows):.4f}")
    print(f"最大值: {np.max(shadows):.4f}")
    print(f"平均值: {np.mean(shadows):.4f}")
    print(f"标准差: {np.std(shadows):.4f}")
    print(f"完全无遮挡的定日镜数量: {np.sum(shadows == 0)}")
    print(f"有遮挡的定日镜数量: {np.sum(shadows > 0)}")
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(location[:, 0], location[:, 1], c=shadows, cmap='viridis', s=50)
    plt.colorbar(label='阴影比例')
    plt.title('定日镜阴影比例分布')
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.hist(shadows, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('阴影比例')
    plt.ylabel('定日镜数量')
    plt.title('阴影比例分布直方图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shadow_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return shadows

if __name__ == "__main__":
    shadows = test_shadow_calculation()
