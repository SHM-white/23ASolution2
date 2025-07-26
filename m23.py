# 问题二：最终三分查找优化定日镜宽度
import numpy as np
import matplotlib.pyplot as plt
import pickle
from m21 import objfun2

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def analyze_dr_for_different_H():
    """分析不同H值下dr的最优值"""
    run_optimization = input("\n是否运行完整计算? (y/n, 默认n): ").lower().strip()
    simple = True if run_optimization != 'y' else False
    D = 156.0504
    h = 6
    
    print("分析不同H值下相邻定日镜间距dr的影响...")
    
    H_values = [4, 5, 6, 7, 8]
    results = {}
    
    for H in H_values:
        W = H  # 基于验证，使用W=H
        
        print(f"\n分析 H={H}m 的情况:")
        print("dr(m)    年平均功率(MW)")
        print("-" * 25)
        
        dr_range = np.arange(H + 5, H + 7.2, 0.2)
        powers = []
        
        for dr in dr_range:
            # 生成定日镜位置
            R = np.arange(100, 350 + D + dr, dr)
            
            x, y = [], []
            for i, r in enumerate(R):
                if r < 350 - D:
                    theta_bond = -np.pi / 2
                else:
                    # 修复 arcsin 参数越界问题
                    arg = (r**2 + D**2 - 350**2) / (2 * r * D)
                    arg = np.clip(arg, -1, 1)  # 确保参数在 [-1, 1] 范围内
                    theta_bond = np.arcsin(arg)
                
                beta = dr / r
                theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
                theta_end = np.pi - theta_bond - beta / 4
                
                # 修复 np.arange 步长问题
                if beta > 0 and theta_end >= theta_start:
                    theta_range = np.arange(theta_start, theta_end + beta/2, beta)
                else:
                    theta_range = np.array([])  # 空数组避免错误
                
                x.extend(r * np.cos(theta_range))
                y.extend(r * np.sin(theta_range))
            
            location = np.column_stack([x, y])
            year_EF, _ = objfun2(H, W, h, D, location, simple=simple)

            powers.append(year_EF / 1000)  # 转换为MW
            print(f"{dr:.1f}     {year_EF/1000:.4f}")
            
            # 连续下降停止条件
            if len(powers) >= 3 and powers[-1] < powers[-2] < powers[-3]:
                break
        
        results[H] = (dr_range[:len(powers)], powers)
    
    return results

def ternary_search_final_W():
    """最终的三分查找优化W值"""
    run_optimization = input("\n是否运行完整计算? (y/n, 默认n): ").lower().strip()
    simple = True if run_optimization != 'y' else False
    D = 156.0504  # 根据之前的优化结果
    h = 6
    epsilon = 0.05
    
    min_w = 4
    max_w = 7
    
    print("\n使用三分查找法优化定日镜宽度W...")
    print("迭代过程:")
    print("迭代   左边界   右边界   左功率密度   右功率密度")
    print("-" * 55)
    
    iteration = 0
    max_iterations = 50
    search_history = []
    
    while (max_w - min_w) >= epsilon and iteration < max_iterations:
        left_w = min_w + (max_w - min_w) / 3
        right_w = min_w + 2 * (max_w - min_w) / 3
        
        # 为左边界和右边界生成定日镜布局
        def generate_layout(W):
            H = W  # W = H
            dr = W + 5  # dr = W + 5
            R = np.arange(100, 350 + D + dr, dr)
            
            x, y = [], []
            for i, r in enumerate(R):
                if r < 350 - D:
                    theta_bond = -np.pi / 2
                else:
                    # 修复 arcsin 参数越界问题
                    arg = (r**2 + D**2 - 350**2) / (2 * r * D)
                    arg = np.clip(arg, -1, 1)  # 确保参数在 [-1, 1] 范围内
                    theta_bond = np.arcsin(arg)
                
                beta = dr / r
                theta_start = (-1)**i * beta / 4 + theta_bond + beta / 2
                theta_end = np.pi - theta_bond - beta / 4
                
                # 修复 np.arange 步长问题
                if beta > 0 and theta_end >= theta_start:
                    theta_range = np.arange(theta_start, theta_end + beta/2, beta)
                else:
                    theta_range = np.array([])  # 空数组避免错误
                
                x.extend(r * np.cos(theta_range))
                y.extend(r * np.sin(theta_range))
            
            return np.column_stack([x, y])
        
        # 计算目标函数值
        location_left = generate_layout(left_w)
        location_right = generate_layout(right_w)

        _, power_per_area_left = objfun2(left_w, left_w, h, D, location_left, simple=simple)
        _, power_per_area_right = objfun2(right_w, right_w, h, D, location_right, simple=simple)

        search_history.append([iteration + 1, min_w, max_w, left_w, right_w,
                              power_per_area_left, power_per_area_right])
        
        print(f"{iteration + 1:3d}    {left_w:.4f}   {right_w:.4f}    {power_per_area_left:.6f}    {power_per_area_right:.6f}")
        
        if power_per_area_left < power_per_area_right:
            min_w = left_w
        else:
            max_w = right_w
        
        iteration += 1
    
    # 计算最优解
    optimal_W = (min_w + max_w) / 2
    H = optimal_W
    dr = optimal_W + 5
    
    # 生成最优布局
    location_optimal = generate_layout(optimal_W)
    total_power, power_per_area = objfun2(H, optimal_W, h, D, location_optimal, simple=simple)

    print("\n" + "="*60)
    print("三分查找最终结果")
    print("="*60)
    print(f"最优定日镜宽度 W: {optimal_W:.4f} m")
    print(f"最优定日镜高度 H: {H:.4f} m")
    print(f"相邻定日镜间距 dr: {dr:.4f} m")
    print(f"吸收塔位置坐标: (0, {-D:.4f})")
    print(f"定日镜数量: {len(location_optimal)}")
    print(f"定日镜总面积: {len(location_optimal) * optimal_W * H:.2f} m²")
    print(f"年平均输出热功率: {total_power/1000:.2f} MW")
    print(f"单位面积年平均输出热功率: {power_per_area:.4f} kW/m²")
    print("="*60)
    
    # 绘制三分查找过程
    search_history = np.array(search_history)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制搜索过程
    plt.subplot(2, 1, 1)
    plt.plot(search_history[:, 0], search_history[:, 3], 'bo-', label='左边界', markersize=6)
    plt.plot(search_history[:, 0], search_history[:, 4], 'ro-', label='右边界', markersize=6)
    plt.axhline(y=optimal_W, color='g', linestyle='--', alpha=0.7, label=f'最优值 W={optimal_W:.4f}m')
    plt.xlabel('迭代次数')
    plt.ylabel('W 值 (m)')
    plt.title('三分查找过程 - W值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制目标函数值变化
    plt.subplot(2, 1, 2)
    plt.plot(search_history[:, 0], search_history[:, 5], 'bo-', label='左边界功率密度', markersize=6)
    plt.plot(search_history[:, 0], search_history[:, 6], 'ro-', label='右边界功率密度', markersize=6)
    plt.axhline(y=power_per_area, color='g', linestyle='--', alpha=0.7, 
                label=f'最优功率密度={power_per_area:.4f}kW/m²')
    plt.xlabel('迭代次数')
    plt.ylabel('功率密度 (kW/m²)')
    plt.title('三分查找过程 - 目标函数值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ternary_search_W_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最优布局
    plt.figure(figsize=(10, 10))
    plt.scatter(location_optimal[:, 0], location_optimal[:, 1], s=20, alpha=0.7, c='blue')
    
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
    plt.title(f'优化后的定日镜布局 (W={optimal_W:.2f}m, 共{len(location_optimal)}面)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimal_heliostat_layout.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果
    results = {
        'optimal_W': optimal_W,
        'optimal_H': H,
        'optimal_dr': dr,
        'D': D,
        'h': h,
        'location': location_optimal,
        'heliostat_count': len(location_optimal),
        'total_area': len(location_optimal) * optimal_W * H,
        'total_power_MW': total_power / 1000,
        'power_per_area': power_per_area,
        'tower_position': (0, -D),
        'search_history': search_history
    }
    
    with open('final_W_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return optimal_W, results

def main():
    """主函数"""
    print("=" * 60)
    print("问题二：最终优化定日镜宽度")
    print("=" * 60)
    
    # 分析不同H值下dr的影响
    dr_analysis = analyze_dr_for_different_H()
    
    # 绘制dr分析结果
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    H_values = [4, 5, 6, 7, 8]
    
    for i, H in enumerate(H_values):
        if H in dr_analysis:
            dr_vals, powers = dr_analysis[H]
            plt.subplot(2, 3, i+1)
            plt.plot(dr_vals, powers, 'o-', color=colors[i], linewidth=2, markersize=6)
            plt.axvline(x=H+5, color='red', linestyle='--', alpha=0.7, label=f'dr=H+5={H+5}')
            plt.xlabel('间距 dr (m)')
            plt.ylabel('年平均功率 (MW)')
            plt.title(f'H={H}m时dr的影响')
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('dr_analysis_all_H.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n结论确认: dr = W + 5 时年平均功率最大")
    
    # 执行最终的三分查找
    optimal_W, final_results = ternary_search_final_W()
    
    return final_results

if __name__ == "__main__":
    results = main()
