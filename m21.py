# 问题二：三分查找优化吸收塔位置
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize_scalar
from m11 import calculate_results

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def objfun2(H, W, h, D, location):
    """
    计算目标函数：年平均输出热功率和单位面积年平均输出热功率
    
    参数:
    H, W, h: 定日镜高度、宽度、安装高度
    D: 吸收塔向南移动距离（正值表示向南移动）
    location: 定日镜位置（相对于原坐标系原点）
    
    返回:
    year_EF: 年平均输出热功率
    year_EF_per_area: 单位面积年平均输出热功率
    """
    s_in, alphas, shade, ntrunc, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI = calculate_results(H, W, location, np.array([0, -D, 80]), h)
    return year_Ef, year_Ef_per_area

# 重写生成定日镜位置函数
def generate_heliostats(D, W):
    """
    生成定日镜位置
    
    参数:
    D: 吸收塔向南移动距离（正值表示向南移动）
    W: 定日镜宽度
    
    返回:
    定日镜位置坐标数组（相对于原坐标系原点(0,0)）
    """
    dr = W + 5  # 相邻定日镜间距
    
    # 吸收塔位置：向南移动D距离，即y坐标为-D
    tower_x, tower_y = 0, -D
    
    # 以吸收塔为中心，生成同心圆布局
    R = np.arange(100, 350 + dr + abs(D), dr)  # 半径范围
    
    x, y = [], []
    # 遍历每个半径
    for i, r in enumerate(R):
        # 计算角度边界，确保定日镜在350m圆形区域内
        # 需要判断以塔为中心、半径为r的圆与350m边界圆的交点
        if D == 0:  # 塔在中心时
            theta_bond = -np.pi / 2
        else:
            # 计算角度限制，确保定日镜不超出350m边界
            # 使用几何关系：塔心到原点距离为D，圆半径为r，边界半径为350
            distance_to_origin = abs(D)
            if r + distance_to_origin <= 350:
                # 整个圆都在边界内
                theta_bond = -np.pi / 2
            else:
                # 部分圆超出边界，需要计算限制角度
                # 使用余弦定理计算角度
                if distance_to_origin + r <= 350:
                    theta_bond = -np.pi / 2
                elif abs(r - distance_to_origin) >= 350:
                    continue  # 整个圆都在边界外，跳过
                else:
                    # 计算交点角度
                    cos_theta = (r**2 + distance_to_origin**2 - 350**2) / (2 * r * distance_to_origin)
                    cos_theta = np.clip(cos_theta, -1, 1)
                    theta_bond = np.arccos(cos_theta)
                    theta_bond = np.pi/2 - theta_bond  # 转换到合适的角度系统
        
        # 放置定日镜
        beta = dr / r  # 角度间隔
        theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
        theta_end = np.pi - theta_bond - beta / 4

        if theta_end > theta_start and beta > 0:
            theta_range = np.arange(theta_start, theta_end, beta)
            
            # 计算相对于吸收塔的极坐标，然后转换为相对于原点的直角坐标
            x_rel_tower = r * np.cos(theta_range)  # 相对于塔的x坐标
            y_rel_tower = r * np.sin(theta_range)  # 相对于塔的y坐标
            
            # 转换为相对于原点(0,0)的坐标
            x.extend(x_rel_tower + tower_x)  # tower_x = 0
            y.extend(y_rel_tower + tower_y)  # tower_y = -D

    return np.column_stack([x, y])

def ternary_search_D():
    """使用三分查找优化吸收塔位置"""
    
    # 三分查找参数
    epsilon = 1.0
    W = 8
    H = 8
    h = 6
    
    min_d = 0
    max_d = 250
    
    
    iteration = 0
    max_iterations = 50
    
    while (max_d - min_d) > epsilon and iteration < max_iterations:
        left_d = min_d + (max_d - min_d) / 3
        right_d = min_d + 2 * (max_d - min_d) / 3
        
        # 计算左点和右点的目标函数值
        location_left = generate_heliostats(left_d, W)
        location_right = generate_heliostats(right_d, W)
        
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
    location_optimal = generate_heliostats(optimal_D, W)
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
    # def generate_heliostats_w(W):
    #     H = W  # 假设宽高相等
    #     dr = W + 5
        
    #     # 吸收塔位置：向南移动D距离，即y坐标为-D
    #     tower_x, tower_y = 0, -D
        
    #     # 以吸收塔为中心，生成同心圆布局
    #     R = np.arange(100, 350 + dr, dr)  # 半径范围
        
    #     x, y = [], []
        
    #     for i, r in enumerate(R):
    #         # 计算角度边界，确保定日镜在350m圆形区域内
    #         if D == 0:  # 塔在中心时
    #             theta_bond = -np.pi / 2
    #         else:
    #             # 计算角度限制，确保定日镜不超出350m边界
    #             distance_to_origin = abs(D)
    #             if r + distance_to_origin <= 350:
    #                 # 整个圆都在边界内
    #                 theta_bond = -np.pi / 2
    #             else:
    #                 # 部分圆超出边界，需要计算限制角度
    #                 if distance_to_origin + r <= 350:
    #                     theta_bond = -np.pi / 2
    #                 elif abs(r - distance_to_origin) >= 350:
    #                     continue  # 整个圆都在边界外，跳过
    #                 else:
    #                     # 计算交点角度
    #                     cos_theta = (r**2 + distance_to_origin**2 - 350**2) / (2 * r * distance_to_origin)
    #                     cos_theta = np.clip(cos_theta, -1, 1)
    #                     theta_bond = np.arccos(cos_theta)
    #                     theta_bond = np.pi/2 - theta_bond
            
    #         beta = dr / r
    #         theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
    #         theta_end = np.pi - theta_bond - beta / 4
            
    #         # 确保有效的角度范围
    #         if theta_end > theta_start and beta > 0:
    #             theta_range = np.arange(theta_start, theta_end, beta)
                
    #             # 计算相对于吸收塔的极坐标，然后转换为相对于原点的直角坐标
    #             x_rel_tower = r * np.cos(theta_range)  # 相对于塔的x坐标
    #             y_rel_tower = r * np.sin(theta_range)  # 相对于塔的y坐标
                
    #             # 转换为相对于原点(0,0)的坐标
    #             x.extend(x_rel_tower + tower_x)  # tower_x = 0
    #             y.extend(y_rel_tower + tower_y)  # tower_y = -D
        
    #     return np.column_stack([x, y])
    
    iteration = 0
    max_iterations = 50
    
    while (max_w - min_w) > epsilon and iteration < max_iterations:
        left_w = min_w + (max_w - min_w) / 3
        right_w = min_w + 2 * (max_w - min_w) / 3
        
        # 计算目标函数值
        location_left = generate_heliostats(D, left_w)
        location_right = generate_heliostats(D, right_w)

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

def visualize_layout(D, location, title="定日镜布局"):
    """可视化定日镜布局"""
    plt.figure(figsize=(12, 10))
    
    # 绘制定日镜位置
    plt.scatter(location[:, 0], location[:, 1], c='blue', s=20, alpha=0.6, label='定日镜')
    
    # 绘制吸收塔位置
    tower_x, tower_y = 0, -D
    plt.scatter(tower_x, tower_y, c='red', s=200, marker='^', label=f'吸收塔 (0, {-D:.1f})')
    
    # 绘制350m边界圆
    circle = plt.Circle((0, 0), 350, fill=False, color='green', linestyle='--', label='350m边界')
    plt.gca().add_patch(circle)
    
    # 绘制以吸收塔为中心的同心圆（验证布局）
    for r in [100, 150, 200, 250, 300]:
        if r <= 350:
            circle_tower = plt.Circle((tower_x, tower_y), r, fill=False, color='gray', alpha=0.3, linestyle=':')
            plt.gca().add_patch(circle_tower)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title(f'{title} (D={D:.1f}m)')
    plt.legend()
    
    # 设置坐标轴范围
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    
    plt.tight_layout()
    plt.show()
    
    # 输出统计信息
    print(f"定日镜总数: {len(location)}")
    print(f"吸收塔位置: (0, {-D:.1f})")
    print(f"定日镜分布范围: X=[{location[:, 0].min():.1f}, {location[:, 0].max():.1f}], Y=[{location[:, 1].min():.1f}, {location[:, 1].max():.1f}]")

def main():
    """主函数"""
    print("开始问题二优化...")
    
    print("\n第一步：优化吸收塔位置...")
    optimal_D, d_results = ternary_search_D()
    
    # 可视化D优化结果
    location_d = generate_heliostats(optimal_D)
    visualize_layout(optimal_D, location_d, "吸收塔位置优化结果")
    
    print("\n第二步：优化定日镜宽度...")
    optimal_W, w_results = ternary_search_W()
    
    # 可视化最终结果
    visualize_layout(w_results['optimal_D'], w_results['location'], "最终优化结果")
    
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
    # 首先测试布局是否正确
    print("测试定日镜布局...")
    
    # 测试不同D值的布局
    test_D_values = [0, 100, 200]
    for D in test_D_values:
        print(f"\n测试 D = {D}:")
        location = generate_heliostats(D)
        print(f"定日镜数量: {len(location)}, 吸收塔位置: (0, {-D})")
    
    print("\n布局测试完成，开始运行完整优化...")
    
    # 运行主优化程序
    results = main()
