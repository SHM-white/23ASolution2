# 问题三：蒙特卡洛方法优化定日镜安装高度
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def objfun3(H, W, h_array, D, location):
    """
    问题三的目标函数：各定日镜高度可变
    
    参数:
    H, W: 定日镜高度、宽度（数组或标量）
    h_array: 各定日镜的安装高度数组
    D: 吸收塔向南移动距离
    location: 定日镜位置
    
    返回:
    year_EF: 年平均输出热功率
    year_EF_per_area: 单位面积年平均输出热功率
    """
    
    # 如果H和W是标量，则扩展为数组
    if np.isscalar(H):
        H = H * np.ones(len(location))
    if np.isscalar(W):
        W = W * np.ones(len(location))
    
    loc_jire = np.array([0, -D, 80])  # 集热器中心坐标
    loc_dingri = np.column_stack([location, h_array])  # 各定日镜中心坐标
    
    # 反射光线方向向量
    s_reflect = loc_jire - loc_dingri
    s_reflect = s_reflect / np.linalg.norm(s_reflect, axis=1, keepdims=True)
    
    # 距离计算
    distances = np.sqrt((location[:, 0] - 0)**2 + (location[:, 1] + D)**2 + h_array**2)
    
    # 高度对效率的影响（较高的定日镜可能有更好的光学性能）
    height_factor = 1.0 + 0.1 * (h_array - np.min(h_array)) / (np.max(h_array) - np.min(h_array))
    
    # 距离因子
    distance_factor = 1.0 / (1.0 + distances / 1000.0)
    
    # 简化的效率计算
    cos_efficiency = 0.8 * distance_factor * height_factor
    
    # 大气透射率
    eta_at = 0.99321 - 0.0001176 * distances + 1.97e-8 * distances**2
    
    # 阴影遮挡效率（考虑高度差异的影响）
    eta_sb = 0.85 * np.ones(len(location))
    # 高度变化对阴影的影响
    for i in range(len(location)):
        nearby_indices = distance.cdist([location[i]], location).flatten().argsort()[1:7]  # 最近的6个
        if len(nearby_indices) > 0:
            height_diff = np.std(h_array[nearby_indices])
            eta_sb[i] = 0.85 + 0.05 * min(height_diff / 2.0, 1.0)  # 高度差异可以减少阴影
    
    # 截断效率
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
    
    alpha_avg = np.pi / 4  # 平均太阳高度角
    DNI = G0 * (a + b * np.exp(-c / np.sin(alpha_avg)))
    
    # 镜面面积
    A = W * H
    
    # 总功率
    total_power = np.sum(DNI * A * eta_total)  # kW
    
    # 单位面积功率
    total_area = np.sum(A)
    power_per_area = total_power / total_area
    
    return total_power, power_per_area

def monte_carlo_optimization():
    """使用蒙特卡洛方法优化定日镜安装高度"""
    
    # 从问题二的结果中获取基本参数
    try:
        with open('final_W_optimization_results.pkl', 'rb') as f:
            base_results = pickle.load(f)
        W = base_results['optimal_W']
        D = base_results['D']
        location = base_results['location']
    except:
        # 如果没有问题二的结果，使用默认值
        W = 5.4670
        D = 237.0240
        # 生成简化的定日镜布局
        dr = W + 5
        R = np.arange(100, 350 + D + dr, dr)
        x, y = [], []
        for i, r in enumerate(R):
            if r < 350 - D:
                theta_bond = -np.pi / 2
            else:
                theta_bond = np.arcsin((r**2 + D**2 - 350**2) / (2 * r * D))
            
            beta = dr / r
            theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
            theta_end = np.pi - theta_bond - beta / 4
            
            theta_range = np.arange(theta_start, theta_end + beta/2, beta)
            
            x.extend(r * np.cos(theta_range))
            y.extend(r * np.sin(theta_range))
        
        location = np.column_stack([x, y])
    
    H = W  # 高度等于宽度
    h0 = 6   # 基准安装高度
    
    # 将定日镜按半径分组（圈数）
    distances_from_center = np.sqrt(location[:, 0]**2 + (location[:, 1] + D)**2)
    dr = W + 5
    circle_indices = (distances_from_center - 100) // dr
    n_circles = int(np.max(circle_indices)) + 1
    
    print(f"定日镜总数: {len(location)}")
    print(f"分为 {n_circles} 圈")
    print(f"每圈的定日镜数量:")
    
    circle_counts = []
    for i in range(n_circles):
        count = np.sum(circle_indices == i)
        circle_counts.append(count)
        print(f"  第{i+1}圈: {count}面")
    
    # 初始化高度分配策略
    step = (6 - W/2) / n_circles  # 高度步长
    h_base = np.arange(W/2 + step/2, 6 + step/2, step)[:n_circles]
    
    print(f"\n初始高度分配 (各圈基准高度):")
    for i in range(n_circles):
        print(f"  第{i+1}圈: {h_base[i]:.2f}m")
    
    # 蒙特卡洛优化
    max_val = 0
    best_h = None
    best_h_circles = None
    optimization_history = []
    
    print(f"\n开始蒙特卡洛优化...")
    print("迭代次数    最佳功率(MW)    当前功率(MW)    单位功率(kW/m²)")
    print("-" * 65)
    
    n_iterations = 100  # 迭代次数
    
    for iteration in range(n_iterations):
        # 随机生成每圈的高度
        h_circles = h_base + step * (np.random.rand(n_circles) - 0.5)
        
        # 确保高度在合理范围内
        h_circles = np.clip(h_circles, W/2, 6)
        
        # 将圈高度分配给各个定日镜
        h_array = np.zeros(len(location))
        for i in range(len(location)):
            circle_idx = int(circle_indices[i])
            h_array[i] = h_circles[circle_idx]
        
        # 计算目标函数
        year_EF, year_EF_per_area = objfun3(H, W, h_array, D, location)
        
        # 更新最优解
        if year_EF > max_val:
            max_val = year_EF
            best_h = h_array.copy()
            best_h_circles = h_circles.copy()
            
            print(f"{iteration+1:8d}    {max_val/1000:8.2f}       {year_EF/1000:8.2f}       {year_EF_per_area:8.4f}")
        
        # 记录优化历史
        if iteration % 100 == 0:
            optimization_history.append([iteration, max_val/1000, year_EF/1000, year_EF_per_area])
    
    # 计算最终结果
    final_power, final_power_per_area = objfun3(H, W, best_h, D, location)
    
    print("\n" + "="*60)
    print("蒙特卡洛优化结果")
    print("="*60)
    print(f"吸收塔位置坐标: (0, {-D:.4f})")
    print(f"定日镜尺寸: {W:.4f}m × {H:.4f}m")
    print(f"定日镜总数量: {len(location)}")
    print(f"定日镜总面积: {len(location) * W * H:.2f} m²")
    print(f"年平均输出热功率: {final_power/1000:.2f} MW")
    print(f"单位面积年平均输出热功率: {final_power_per_area:.4f} kW/m²")
    
    print(f"\n各圈最优安装高度:")
    for i in range(n_circles):
        print(f"  第{i+1}圈: {best_h_circles[i]:.2f}m (包含{circle_counts[i]}面定日镜)")
    
    print("="*60)
    
    # 绘制优化过程
    optimization_history = np.array(optimization_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制功率优化过程
    ax1.plot(optimization_history[:, 0], optimization_history[:, 1], 'b-', linewidth=2, label='最佳功率')
    ax1.plot(optimization_history[:, 0], optimization_history[:, 2], 'r-', alpha=0.6, label='当前功率')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('年平均功率 (MW)')
    ax1.set_title('蒙特卡洛优化过程 - 功率变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制单位功率优化过程  
    ax2.plot(optimization_history[:, 0], optimization_history[:, 3], 'g-', linewidth=2)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('单位面积功率 (kW/m²)')
    ax2.set_title('蒙特卡洛优化过程 - 单位面积功率变化')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_optimization_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最优高度分布
    plt.figure(figsize=(12, 8))
    
    # 散点图显示定日镜位置和高度
    scatter = plt.scatter(location[:, 0], location[:, 1], c=best_h, cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, label='安装高度 (m)')
    
    # 绘制镜场边界
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 350 * np.cos(theta)
    circle_y = 350 * np.sin(theta) - D
    plt.plot(circle_x, circle_y, 'r--', linewidth=2, label='镜场边界')
    
    # 绘制吸收塔禁区
    forbidden_x = 100 * np.cos(theta)
    forbidden_y = 100 * np.sin(theta) - D
    plt.plot(forbidden_x, forbidden_y, 'r-', linewidth=2, label='吸收塔禁区')
    
    # 标记吸收塔位置
    plt.plot(0, -D, 'rs', markersize=10, label='吸收塔')
    
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('优化后的定日镜高度分布')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimal_height_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制各圈高度条形图
    plt.figure(figsize=(10, 6))
    circles = np.arange(1, n_circles + 1)
    bars = plt.bar(circles, best_h_circles, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # 在每个条形上标注数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}m\n({circle_counts[i]}面)',
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('圈数')
    plt.ylabel('安装高度 (m)')
    plt.title('各圈定日镜最优安装高度')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(circles)
    plt.ylim(0, 7)
    plt.savefig('optimal_heights_by_circle.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果
    results = {
        'W': W,
        'H': H,
        'D': D,
        'location': location,
        'best_h': best_h,
        'best_h_circles': best_h_circles,
        'circle_indices': circle_indices,
        'circle_counts': circle_counts,
        'n_circles': n_circles,
        'heliostat_count': len(location),
        'total_area': len(location) * W * H,
        'final_power_MW': final_power / 1000,
        'final_power_per_area': final_power_per_area,
        'tower_position': (0, -D),
        'optimization_history': optimization_history
    }
    
    with open('monte_carlo_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def compare_with_problem2():
    """与问题二结果进行比较"""
    
    try:
        # 加载问题二结果
        with open('final_W_optimization_results.pkl', 'rb') as f:
            problem2_results = pickle.load(f)
        
        # 加载问题三结果
        with open('monte_carlo_optimization_results.pkl', 'rb') as f:
            problem3_results = pickle.load(f)
        
        print("\n" + "="*60)
        print("问题二与问题三结果对比")
        print("="*60)
        print(f"{'项目':<25} {'问题二':<15} {'问题三':<15} {'改进':<10}")
        print("-" * 65)
        
        power2 = problem2_results['total_power_MW']
        power3 = problem3_results['final_power_MW']
        power_improvement = ((power3 - power2) / power2) * 100
        
        area_power2 = problem2_results['power_per_area']
        area_power3 = problem3_results['final_power_per_area']
        area_improvement = ((area_power3 - area_power2) / area_power2) * 100
        
        print(f"{'年平均功率(MW)':<25} {power2:<15.2f} {power3:<15.2f} {power_improvement:>+8.2f}%")
        print(f"{'单位面积功率(kW/m²)':<25} {area_power2:<15.4f} {area_power3:<15.4f} {area_improvement:>+8.2f}%")
        print(f"{'定日镜数量':<25} {problem2_results['heliostat_count']:<15d} {problem3_results['heliostat_count']:<15d}")
        print(f"{'高度策略':<25} {'统一6m':<15} {'各圈可变':<15}")
        
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"无法加载结果文件: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("问题三：蒙特卡洛方法优化定日镜安装高度")
    print("=" * 60)
    
    # 执行蒙特卡洛优化
    results = monte_carlo_optimization()
    
    # 与问题二进行比较
    compare_with_problem2()
    
    return results

if __name__ == "__main__":
    results = main()
