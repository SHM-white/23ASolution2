# 问题一：计算各个效率
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def load_results():
    """加载问题一的计算结果"""
    with open('Q1_results_optimized.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_efficiencies():
    """计算各种效率"""
    data = load_results()
    location = data['location']
    s_in = data['s_in']
    shade = data['shade']
    ntrunc = data['ntrunc']
    s_reflect = data['s_reflect']
    
    # 定日镜参数
    W, H, h = 6, 6, 4
    xid, yid = int(W/1.2), int(H/1.2)  # 网格数
    loc_jire = np.array([0, 0, 80])
    
    # 1. 余弦效率
    eta_cos = np.zeros((12, 5, len(location)))
    
    for i in range(12):  # 月份
        for j in range(5):  # 时刻
            # 确定每个定日镜的法向量
            n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
            n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
            
            for k in range(len(location)):
                eta_cos[i, j, k] = abs(np.dot(n_dingri[k], s_in[i, 3*j:3*j+3]))
    
    # 计算月平均和年平均余弦效率
    month_eta_cos = np.zeros(12)
    for i in range(12):
        month_eta_cos[i] = np.mean(eta_cos[i, :, :])
    year_eta_cos = np.mean(month_eta_cos)
    
    # 绘制散点图
    tmp = np.mean(eta_cos, axis=(0, 1))
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=tmp, cmap='viridis')
    plt.title('每块定日镜年平均余弦效率散点图')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('cosine_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 阴影遮挡效率
    eta_sb = 1 - shade
    
    # 计算月平均和年平均阴影遮挡效率
    month_eta_sb = np.zeros(12)
    for i in range(12):
        month_eta_sb[i] = np.mean(eta_sb[i, :, :])
    year_eta_sb = np.mean(month_eta_sb)
    
    # 绘制散点图
    tmp = np.mean(eta_sb, axis=(0, 1))
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=tmp, cmap='viridis')
    plt.title('每块定日镜年平均阴影遮挡效率散点图')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('shadow_blocking_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 大气透射率
    tmp = location[:, 0]**2 + location[:, 1]**2 + (h - loc_jire[2])**2
    d_HR = np.sqrt(tmp)
    tmp1 = 0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR**2
    eta_at = np.zeros((12, 5, len(location)))
    
    for i in range(len(location)):
        eta_at[:, :, i] = tmp1[i]
    
    # 绘制散点图
    tmp = np.mean(eta_at, axis=(0, 1))
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=tmp, cmap='viridis')
    plt.title('每块定日镜大气透射率散点图')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('atmospheric_transmittance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 截断效率
    eta_trunc = ntrunc.copy()
    
    # 处理无穷值
    inf_indices = np.where(eta_trunc == np.inf)
    valid_values = eta_trunc[eta_trunc != np.inf]
    if len(valid_values) > 0:
        ave = np.mean(valid_values)
        eta_trunc[inf_indices] = ave
    
    # 计算月平均和年平均截断效率
    month_eta_trunc = np.zeros(12)
    for i in range(12):
        tmp = eta_trunc[i, :, :]
        month_eta_trunc[i] = np.mean(tmp[tmp != np.inf])
    year_eta_trunc = np.mean(month_eta_trunc)
    
    # 绘制散点图
    tmp = np.mean(eta_trunc, axis=(0, 1))
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=tmp, cmap='viridis')
    plt.title('每块定日镜年平均截断效率散点图')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('truncation_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 镜面反射率
    eta_ref = 0.92 * np.ones((12, 5, len(location)))
    
    # 6. 光学效率
    eta = eta_sb * eta_cos * eta_at * eta_trunc * eta_ref
    
    # 计算月平均和年平均光学效率
    month_eta = np.zeros(12)
    for i in range(12):
        month_eta[i] = np.mean(eta[i, :, :])
    year_eta = np.mean(month_eta)
    
    # 绘制散点图
    tmp = np.mean(eta, axis=(0, 1))
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=tmp, cmap='viridis')
    plt.title('每块定日镜年平均光学效率散点图')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('optical_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 单位面积镜面输出热功率
    # 构造DNI矩阵
    G0 = 1.366
    altitude = 3
    a = 0.4237 - 0.00821 * (6 - altitude)**2
    b = 0.5055 + 0.00595 * (6.5 - altitude)**2
    c = 0.2711 + 0.01858 * (2.5 - altitude)**2
    
    # 从之前保存的alphas数据计算DNI
    with open('Q1_results.pkl', 'rb') as f:
        data = pickle.load(f)
    alphas = data['alphas']
    
    tmp2 = G0 * (a + b * np.exp(-c / np.sin(alphas)))  # 12x5矩阵
    DNI = np.zeros((12, 5, len(location)))
    for i in range(len(location)):
        DNI[:, :, i] = tmp2
    
    # 定日镜面积
    A = W * H * np.ones((12, 5, len(location)))
    E = DNI * A * eta  # 每块热功率，12x5xN
    
    Ef = np.sum(E, axis=2)  # 镜场总瞬时热功率，12x5
    year_Ef = np.mean(Ef)  # 镜场年平均热功率
    
    tmp_total_area = np.sum(A, axis=2)
    Ef_per_area = Ef / tmp_total_area[0, 0]  # 单位面积输出热功率，12x5
    
    # 每块镜子年平均热功率
    mirror_year_ave = np.mean(E, axis=(0, 1))
    
    # 单位面积输出月、年平均
    month_Ef_per_area = np.mean(Ef_per_area, axis=1)
    year_Ef_per_area = np.mean(month_Ef_per_area)
    
    # 每块定日镜年平均输出热功率散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(location[:, 0], location[:, 1], c=mirror_year_ave, cmap='viridis')
    plt.title('每块定日镜年平均输出热功率散点图(kW)')
    plt.colorbar()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.axis('equal')
    plt.savefig('heat_power_output.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出结果
    print("=" * 60)
    print("问题一：定日镜场光学效率与输出热功率计算结果")
    print("=" * 60)
    print(f"年平均光学效率: {year_eta:.4f}")
    print(f"年平均余弦效率: {year_eta_cos:.4f}")
    print(f"年平均阴影遮挡效率: {year_eta_sb:.4f}")
    print(f"年平均截断效率: {year_eta_trunc:.4f}")
    print(f"年平均输出热功率: {year_Ef/1000:.2f} MW")
    print(f"单位面积镜面年平均输出热功率: {year_Ef_per_area:.4f} kW/m²")
    print("=" * 60)
    
    # 创建月度结果表格
    months = ['1月21日', '2月21日', '3月21日', '4月21日', '5月21日', '6月21日',
              '7月21日', '8月21日', '9月21日', '10月21日', '11月21日', '12月21日']
    
    print("\n每月21日平均光学效率及输出功率:")
    print(f"{'日期':<8} {'平均光学效率':<12} {'平均余弦效率':<12} {'平均阴影遮挡效率':<15} {'平均截断效率':<12} {'单位面积镜面平均输出热功率(kW/m²)':<25}")
    print("-" * 100)
    
    for i in range(12):
        print(f"{months[i]:<8} {month_eta[i]:<12.4f} {month_eta_cos[i]:<12.4f} "
              f"{month_eta_sb[i]:<15.4f} {month_eta_trunc[i]:<12.4f} {month_Ef_per_area[i]:<25.4f}")
    
    # 保存计算结果
    results = {
        'year_eta': year_eta,
        'year_eta_cos': year_eta_cos,
        'year_eta_sb': year_eta_sb,
        'year_eta_trunc': year_eta_trunc,
        'year_Ef': year_Ef,
        'year_Ef_per_area': year_Ef_per_area,
        'month_eta': month_eta,
        'month_eta_cos': month_eta_cos,
        'month_eta_sb': month_eta_sb,
        'month_eta_trunc': month_eta_trunc,
        'month_Ef_per_area': month_Ef_per_area,
        'mirror_year_ave': mirror_year_ave
    }
    
    with open('Q1_efficiency_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    results = calculate_efficiencies()
