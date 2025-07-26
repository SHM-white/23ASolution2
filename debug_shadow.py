# 详细调试阴影计算
import numpy as np
from m11_optimized import OptimizedShadowCalculator, calc_plane_line_intersect_point, is_point_in_rectangular, read_data

def debug_shadow_calculation():
    """详细调试阴影计算过程"""
    print("详细调试阴影计算...")
    
    # 使用实际的定日镜场数据
    location = read_data()
    
    # 选择两个相邻的定日镜进行测试
    distances = []
    for i in range(min(50, len(location))):
        for j in range(i+1, min(50, len(location))):
            dist = np.linalg.norm(location[i] - location[j])
            distances.append((dist, i, j))
    
    # 找到最近的一对定日镜
    distances.sort()
    _, k1, k2 = distances[0]
    
    test_location = np.array([location[k1], location[k2]])
    print(f"选择定日镜 {k1} 和 {k2} 进行测试")
    print(f"位置: {test_location}")
    print(f"距离: {np.linalg.norm(test_location[0] - test_location[1]):.2f}m")
    
    calculator = OptimizedShadowCalculator(test_location, 6, 6, 4)
    
    # 使用实际的太阳参数（12月21日，上午9点）
    D = 334  # 12月21日
    ST = 9.0  # 上午9点
    phi = 39.4 * np.pi / 180  # 当地纬度
    
    # 计算太阳位置
    delta = np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360))
    w = (ST - 12) * np.pi / 12
    alpha = np.arcsin(np.cos(delta) * np.cos(phi) * np.cos(w) + np.sin(delta) * np.sin(phi))
    
    cos_gamma = (np.sin(delta) - np.sin(alpha) * np.sin(phi)) / (np.cos(alpha) * np.cos(phi))
    cos_gamma = np.clip(cos_gamma, -1, 1)  # 防止数值错误
    gamma = np.arccos(cos_gamma)
    
    # 上午时间，太阳在东边，方位角为负
    if w < 0:
        gamma = -gamma
    
    print(f"太阳高度角: {alpha * 180 / np.pi:.2f}度")
    print(f"太阳方位角: {gamma * 180 / np.pi:.2f}度")
    
    # 入射光方向向量（从太阳指向地面）
    s_in_current = -np.array([np.sin(gamma), np.cos(gamma), -np.tan(alpha)])
    s_in_current = s_in_current / np.linalg.norm(s_in_current)
    
    print(f"太阳方向向量: {s_in_current}")
    
    # 反射光方向（向集热器）
    loc_jire = np.array([0, 0, 80])
    h = 4
    s_reflect = np.zeros((2, 3))
    for i in range(2):
        s_reflect[i] = loc_jire - np.array([test_location[i, 0], test_location[i, 1], h])
        s_reflect[i] = s_reflect[i] / np.linalg.norm(s_reflect[i])
    
    print(f"反射光方向向量: {s_reflect}")
    
    # 计算法向量（按照MATLAB的方式）
    n_dingri = np.zeros((2, 3))
    for i in range(2):
        n_dingri[i] = s_in_current - s_reflect[i]
        n_dingri[i] = n_dingri[i] / np.linalg.norm(n_dingri[i])
    
    print(f"法向量: {n_dingri}")
    
    # 计算角点（按照MATLAB的方式）
    v1 = np.column_stack([n_dingri[:, 1], -n_dingri[:, 0], np.zeros(n_dingri.shape[0])])
    v2 = np.column_stack([
        -n_dingri[:, 0] * n_dingri[:, 2],
        -n_dingri[:, 1] * n_dingri[:, 2], 
        n_dingri[:, 1]**2 + n_dingri[:, 0]**2
    ])
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    
    # 计算角点
    W, H = 6, 6
    corners = np.zeros((2, 4, 3))
    
    corners[:, 0] = np.column_stack([
        test_location[:, 0] + W * v1[:, 0] / 2 + H * v2[:, 0] / 2,
        test_location[:, 1] + W * v1[:, 1] / 2 + H * v2[:, 1] / 2,
        h + W * v1[:, 2] / 2 + H * v2[:, 2] / 2
    ])
    
    corners[:, 1] = np.column_stack([
        test_location[:, 0] - W * v1[:, 0] / 2 + H * v2[:, 0] / 2,
        test_location[:, 1] - W * v1[:, 1] / 2 + H * v2[:, 1] / 2,
        h - W * v1[:, 2] / 2 + H * v2[:, 2] / 2
    ])
    
    corners[:, 2] = np.column_stack([
        test_location[:, 0] - W * v1[:, 0] / 2 - H * v2[:, 0] / 2,
        test_location[:, 1] - W * v1[:, 1] / 2 - H * v2[:, 1] / 2,
        h - W * v1[:, 2] / 2 - H * v2[:, 2] / 2
    ])
    
    corners[:, 3] = np.column_stack([
        test_location[:, 0] + W * v1[:, 0] / 2 - H * v2[:, 0] / 2,
        test_location[:, 1] + W * v1[:, 1] / 2 - H * v2[:, 1] / 2,
        h + W * v1[:, 2] / 2 - H * v2[:, 2] / 2
    ])
    
    time_data = {
        'n_dingri': n_dingri,
        'v1': v1,
        'v2': v2,
        'corners': corners,
        's_in_current': s_in_current
    }
    
    # 计算两个定日镜的阴影
    for k in range(2):
        print(f"\n=== 计算定日镜{k}的阴影 ===")
        shadow_ratio = calculator.calculate_single_heliostat_shadow(k, time_data)
        print(f"最终阴影比例: {shadow_ratio:.4f}")
        
        # 详细分析一个网格点
        v1_k = v1[k]
        v2_k = v2[k]
        corner1_k = corners[k, 1]  # xp2对应的角点
        
        # 选择中心网格点
        dl = H / 5
        ii, jj = 2, 2  # 中心网格
        xi = corner1_k[0] + jj * dl * v1_k[0] - ii * dl * v2_k[0] - dl * v1_k[0] / 2 + dl * v2_k[0] / 2
        yi = corner1_k[1] + jj * dl * v1_k[1] - ii * dl * v2_k[1] - dl * v1_k[1] / 2 + dl * v2_k[1] / 2
        zi = corner1_k[2] + jj * dl * v1_k[2] - ii * dl * v2_k[2] - dl * v1_k[2] / 2 + dl * v2_k[2] / 2
        
        print(f"  中心网格点: ({xi:.2f}, {yi:.2f}, {zi:.2f})")
        
        # 检查是否被另一个定日镜遮挡
        kk = 1 - k  # 另一个定日镜
        print(f"  检查是否被定日镜{kk}遮挡")
        
        # 计算交点
        px1, py1, pz1 = calc_plane_line_intersect_point(
            n_dingri[kk], [test_location[kk, 0], test_location[kk, 1], h],
            s_in_current, [xi, yi, zi]
        )
        
        if px1 is not None:
            print(f"  入射光线交点: ({px1:.2f}, {py1:.2f}, {pz1:.2f})")
            
            # 检查点积条件
            heliostat_center = np.array([test_location[kk, 0], test_location[kk, 1], h])
            intersection_point = np.array([px1, py1, pz1])
            dot_product = np.dot(heliostat_center - intersection_point, s_in_current)
            
            print(f"  点积条件: {dot_product:.6f} > 0? {dot_product > 0}")
            
            if dot_product > 0:
                in_rect = is_point_in_rectangular(px1, py1, pz1, corners[kk])
                print(f"  交点在矩形内? {in_rect}")
                
                if not in_rect:
                    print(f"  定日镜{kk}的角点:")
                    for i in range(4):
                        print(f"    角点{i}: ({corners[kk, i, 0]:.2f}, {corners[kk, i, 1]:.2f}, {corners[kk, i, 2]:.2f})")
        else:
            print(f"  没有找到入射光交点")

if __name__ == "__main__":
    debug_shadow_calculation()
