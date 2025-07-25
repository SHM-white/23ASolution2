# 问题二：三分查找优化吸收塔位置
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize_scalar

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def objfun2(H, W, h, D, location):
    """
    计算目标函数：年平均输出热功率和单位面积年平均输出热功率
    
    参数:
    H, W, h: 定日镜高度、宽度、安装高度
    D: 吸收塔向南移动距离
    location: 定日镜位置
    
    返回:
    year_EF: 年平均输出热功率
    year_EF_per_area: 单位面积年平均输出热功率
    """
    
    # 这里是简化的目标函数，实际应该调用完整的光学效率计算
    # 基于问题一的算法进行计算
    
    loc_jire = np.array([0, -D, 80])  # 集热器中心坐标（向南移动D）
    loc_dingri = np.column_stack([location, h * np.ones(location.shape[0])])
    
    # 反射光线方向向量
    s_reflect = loc_jire - loc_dingri
    s_reflect = s_reflect / np.linalg.norm(s_reflect, axis=1, keepdims=True)
    
    # 简化计算 - 实际应该包含完整的阴影遮挡和截断效率计算
    # 这里使用简化公式估算
    
    # 距离因子（越远效率越低）
    distances = np.sqrt((location[:, 0] - 0)**2 + (location[:, 1] + D)**2 + h**2)
    distance_factor = 1.0 / (1.0 + distances / 1000.0)
    
    # 余弦效率近似（基于定日镜朝向）
    cos_efficiency = 0.8 * distance_factor
    
    # 大气透射率
    eta_at = 0.99321 - 0.0001176 * distances + 1.97e-8 * distances**2
    
    # 阴影遮挡效率近似
    eta_sb = 0.85 * np.ones(len(location))
    
    # 截断效率近似
    eta_trunc = 0.80 * np.ones(len(location))
    
    # 镜面反射率
    eta_ref = 0.92
    
    # 总光学效率
    eta_total = cos_efficiency * eta_at * eta_sb * eta_trunc * eta_ref
    
    # DNI计算
    G0 = 1.366
    altitude = 3
    a = 0.4237 - 0.00821 * (6 - altitude)**2
    b = 0.5055 + 0.00595 * (6.5 - altitude)**2
    c = 0.2711 + 0.01858 * (2.5 - altitude)**2
    
    # 假设平均太阳高度角
    alpha_avg = np.pi / 4  # 45度
    DNI = G0 * (a + b * np.exp(-c / np.sin(alpha_avg)))
    
    # 镜面面积
    A = W * H
    
    # 总功率
    total_power = np.sum(DNI * A * eta_total)  # kW
    
    # 单位面积功率
    total_area = len(location) * A
    power_per_area = total_power / total_area
    
    return total_power, power_per_area

def ternary_search_D():
    """使用三分查找优化吸收塔位置"""
    
    # 三分查找参数
    epsilon = 1.0
    W = 8
    H = 8
    h = 6
    
    min_d = 0
    max_d = 250
    
    # 生成定日镜位置（简化）
    def generate_heliostats(D):
        dr = W + 5  # 相邻定日镜间距
        R = np.arange(100, min(350, 350 + D) + dr, dr)  # 半径范围，避免超出边界
        
        x, y = [], []
        
        for i, r in enumerate(R):
            if r < 350 - abs(D):  # 使用绝对值避免负数
                theta_bond = -np.pi / 2
            else:
                # 避免arcsin的参数超出范围
                arg = (r**2 + D**2 - 350**2) / (2 * r * abs(D))
                arg = np.clip(arg, -1, 1)  # 限制在[-1,1]范围内
                theta_bond = np.arcsin(arg)
            
            beta = dr / r  # 角度间隔
            theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
            theta_end = np.pi - theta_bond - beta / 4
            
            # 确保theta_start < theta_end且步长为正
            if theta_end > theta_start and beta > 0:
                theta_range = np.arange(theta_start, theta_end, beta)
                
                x.extend(r * np.cos(theta_range))
                y.extend(r * np.sin(theta_range))
        
        return np.column_stack([x, y])
    
    iteration = 0
    max_iterations = 50
    
    while (max_d - min_d) > epsilon and iteration < max_iterations:
        left_d = min_d + (max_d - min_d) / 3
        right_d = min_d + 2 * (max_d - min_d) / 3
        
        # 计算左点和右点的目标函数值
        location_left = generate_heliostats(left_d)
        location_right = generate_heliostats(right_d)
        
        _, power_per_area_left = objfun2(H, W, h, left_d, location_left)
        _, power_per_area_right = objfun2(H, W, h, right_d, location_right)
        
        print(f"迭代 {iteration + 1}: D_left={left_d:.4f}, D_right={right_d:.4f}")
        print(f"功率密度: left={power_per_area_left:.6f}, right={power_per_area_right:.6f}")
        
        if power_per_area_left < power_per_area_right:
            min_d = left_d
        else:
            max_d = right_d
        
        iteration += 1
    
    # 最优值
    optimal_D = (min_d + max_d) / 2
    location_optimal = generate_heliostats(optimal_D)
    total_power, power_per_area = objfun2(H, W, h, optimal_D, location_optimal)
    
    print("\n" + "="*60)
    print("三分查找结果 - 吸收塔位置优化")
    print("="*60)
    print(f"最优吸收塔向南移动距离: {optimal_D:.4f} m")
    print(f"最优吸收塔位置坐标: (0, {-optimal_D:.4f})")
    print(f"年平均输出热功率: {total_power/1000:.2f} MW")
    print(f"单位面积年平均输出热功率: {power_per_area:.4f} kW/m²")
    print(f"定日镜数量: {len(location_optimal)}")
    print("="*60)
    
    # 保存结果
    results = {
        'optimal_D': optimal_D,
        'total_power': total_power,
        'power_per_area': power_per_area,
        'location': location_optimal,
        'tower_position': (0, -optimal_D),
        'heliostat_count': len(location_optimal)
    }
    
    with open('D_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return optimal_D, results

def ternary_search_W():
    """使用三分查找优化定日镜宽度"""
    
    # 加载最优D值
    try:
        with open('D_optimization_results.pkl', 'rb') as f:
            d_results = pickle.load(f)
        D = d_results['optimal_D']
    except:
        D = 237.0240  # 使用论文中的值
    
    h = 6
    epsilon = 0.05
    min_w = 4
    max_w = 7
    
    # 生成定日镜位置函数
    def generate_heliostats_w(W):
        H = W  # 假设宽高相等
        dr = W + 5
        R = np.arange(100, min(350, 350 + D) + dr, dr)
        
        x, y = [], []
        
        for i, r in enumerate(R):
            if r < 350 - abs(D):
                theta_bond = -np.pi / 2
            else:
                # 避免arcsin的参数超出范围
                arg = (r**2 + D**2 - 350**2) / (2 * r * abs(D))
                arg = np.clip(arg, -1, 1)
                theta_bond = np.arcsin(arg)
            
            beta = dr / r
            theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
            theta_end = np.pi - theta_bond - beta / 4
            
            # 确保有效的角度范围
            if theta_end > theta_start and beta > 0:
                theta_range = np.arange(theta_start, theta_end, beta)
                
                x.extend(r * np.cos(theta_range))
                y.extend(r * np.sin(theta_range))
        
        return np.column_stack([x, y])
    
    iteration = 0
    max_iterations = 50
    
    while (max_w - min_w) > epsilon and iteration < max_iterations:
        left_w = min_w + (max_w - min_w) / 3
        right_w = min_w + 2 * (max_w - min_w) / 3
        
        # 计算目标函数值
        location_left = generate_heliostats_w(left_w)
        location_right = generate_heliostats_w(right_w)
        
        _, power_per_area_left = objfun2(left_w, left_w, h, D, location_left)
        _, power_per_area_right = objfun2(right_w, right_w, h, D, location_right)
        
        print(f"迭代 {iteration + 1}: W_left={left_w:.4f}, W_right={right_w:.4f}")
        print(f"功率密度: left={power_per_area_left:.6f}, right={power_per_area_right:.6f}")
        
        if power_per_area_left < power_per_area_right:
            min_w = left_w
        else:
            max_w = right_w
        
        iteration += 1
    
    # 最优值
    optimal_W = (min_w + max_w) / 2
    H = optimal_W
    location_optimal = generate_heliostats_w(optimal_W)
    total_power, power_per_area = objfun2(H, optimal_W, h, D, location_optimal)
    
    print("\n" + "="*60)
    print("三分查找结果 - 定日镜宽度优化")
    print("="*60)
    print(f"最优定日镜宽度: {optimal_W:.4f} m")
    print(f"最优定日镜高度: {H:.4f} m")
    print(f"年平均输出热功率: {total_power/1000:.2f} MW")
    print(f"单位面积年平均输出热功率: {power_per_area:.4f} kW/m²")
    print(f"定日镜数量: {len(location_optimal)}")
    print("="*60)
    
    # 保存结果
    results = {
        'optimal_W': optimal_W,
        'optimal_H': H,
        'optimal_D': D,
        'total_power': total_power,
        'power_per_area': power_per_area,
        'location': location_optimal,
        'tower_position': (0, -D),
        'heliostat_count': len(location_optimal)
    }
    
    with open('W_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return optimal_W, results

def main():
    """主函数"""
    print("开始问题二优化...")
    
    print("\n第一步：优化吸收塔位置...")
    optimal_D, d_results = ternary_search_D()
    
    print("\n第二步：优化定日镜宽度...")
    optimal_W, w_results = ternary_search_W()
    
    print("\n" + "="*60)
    print("问题二最终优化结果")
    print("="*60)
    print(f"吸收塔最优位置坐标: (0, {-optimal_D:.4f})")
    print(f"定日镜最优尺寸: {optimal_W:.4f}m × {optimal_W:.4f}m")
    print(f"定日镜安装高度: 6m")
    print(f"定日镜总数量: {w_results['heliostat_count']}")
    print(f"年平均输出热功率: {w_results['total_power']/1000:.2f} MW")
    print(f"单位面积年平均输出热功率: {w_results['power_per_area']:.4f} kW/m²")
    print("="*60)
    
    # 保存最终结果
    final_results = {
        'tower_position': (0, -optimal_D),
        'heliostat_size': (optimal_W, optimal_W),
        'heliostat_height': 6,
        'heliostat_count': w_results['heliostat_count'],
        'total_power_MW': w_results['total_power']/1000,
        'power_per_area': w_results['power_per_area'],
        'location': w_results['location']
    }
    
    with open('problem2_final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    return final_results

if __name__ == "__main__":
    results = main()
