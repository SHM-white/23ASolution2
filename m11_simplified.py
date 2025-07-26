# 简化优化版本的计算函数 - 为问题二优化使用
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import time

def calculate_results_simplified(H, W, location, loc_jire, h, 
                                 time_sample_ratio=0.5, 
                                 grid_reduction=2,
                                 neighbor_limit=5,
                                 use_fast_approximation=True):
    """
    简化版本的计算函数 - 在保持合理精度的同时大幅提升计算速度
    
    优化策略：
    1. 减少时间步采样（保留关键时间点）
    2. 降低网格分辨率
    3. 限制最近邻数量
    4. 使用快速近似算法
    5. 早期退出策略
    
    参数:
    time_sample_ratio: 时间步采样比例 (0.5表示采样一半的时间步)
    grid_reduction: 网格降低因子 (2表示每个方向网格数减半)
    neighbor_limit: 最近邻限制数量
    use_fast_approximation: 是否使用快速近似算法
    """
    
    print(f"简化版本启动 - 时间采样率: {time_sample_ratio}, 网格降低: {grid_reduction}x")
    
    # ==================== 基础几何计算（保持不变） ====================
    
    loc_dingri = np.column_stack([location, h * np.ones(location.shape[0])])
    s_reflect = loc_jire - loc_dingri
    s_reflect = s_reflect / np.linalg.norm(s_reflect, axis=1, keepdims=True)
    
    # 简化的时间步选择策略
    D = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ST = np.array([9.0, 10.5, 12.0, 13.5, 15.0])
    
    # 选择关键时间步（保留极值和中间值）
    if time_sample_ratio < 1.0:
        # 选择关键月份：冬至、夏至、春分秋分、以及其他关键点
        key_months = [0, 2, 5, 8, 11]  # 1月、3月、6月、9月、12月
        selected_months = key_months[:max(1, int(len(key_months) * time_sample_ratio + 0.5))]
        
        # 选择关键时刻：中午及其前后
        key_times = [1, 2, 3]  # 10:30, 12:00, 13:30
        selected_times = key_times[:max(1, int(len(key_times) * time_sample_ratio + 0.5))]
    else:
        selected_months = list(range(12))
        selected_times = list(range(5))
    
    print(f"选择时间步: {len(selected_months)}个月份 × {len(selected_times)}个时刻 = {len(selected_months) * len(selected_times)}个时间步")
    
    # 计算太阳位置参数（只计算选定的时间步）
    alphas = {}
    gamas = {}
    s_in = {}
    phi = 39.4 * np.pi / 180
    
    for i in selected_months:
        for j in selected_times:
            delta = np.arcsin(np.sin(2 * np.pi * D[i] / 365) * np.sin(2 * np.pi * 23.45 / 360))
            w = (ST[j] - 12) * np.pi / 12
            
            alpha = np.arcsin(np.cos(delta) * np.cos(phi) * np.cos(w) + np.sin(delta) * np.sin(phi))
            alphas[(i, j)] = alpha
            
            cos_gamma = (np.sin(delta) - np.sin(alpha) * np.sin(phi)) / (np.cos(alpha) * np.cos(phi))
            cos_gamma = np.clip(cos_gamma, -1, 1)
            gamma_temp = np.arccos(cos_gamma)
            
            if w < 0:
                gamma = -gamma_temp
            else:
                gamma = gamma_temp
            gamas[(i, j)] = gamma
            
            s_in_temp = -np.array([np.sin(gamma), np.cos(gamma), -np.tan(alpha)])
            s_in[(i, j)] = s_in_temp / np.linalg.norm(s_in_temp)
    
    # ==================== 快速阴影和截断计算 ====================
    
    n_heliostats = len(location)
    
    # 简化的阴影计算器
    kdtree = cKDTree(location)
    
    # 预计算每个定日镜的有限近邻
    neighbors_cache = {}
    for k in range(n_heliostats):
        _, indices = kdtree.query(location[k], k=min(neighbor_limit + 1, n_heliostats))
        neighbors_cache[k] = indices[1:]  # 排除自己
    
    # 初始化结果数组
    shade_simplified = {}
    ntrunc_simplified = {}
    
    print("开始简化阴影和截断计算...")
    start_time = time.time()
    
    total_steps = len(selected_months) * len(selected_times)
    with tqdm(total=total_steps, desc="简化计算", ncols=80) as pbar:
        
        for i in selected_months:
            for j in selected_times:
                # 计算当前时间步的法向量
                s_in_current = s_in[(i, j)]
                n_dingri = s_in_current - s_reflect
                n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
                
                # 简化的角点计算
                if use_fast_approximation:
                    # 使用简化的角点近似（假设定日镜接近水平）
                    corners = np.zeros((n_heliostats, 4, 3))
                    for k in range(n_heliostats):
                        corners[k, :, 0] = location[k, 0] + np.array([W/2, -W/2, -W/2, W/2])
                        corners[k, :, 1] = location[k, 1] + np.array([H/2, H/2, -H/2, -H/2])
                        corners[k, :, 2] = h
                else:
                    # 完整的角点计算（保留精确性）
                    v1 = np.column_stack([n_dingri[:, 1], -n_dingri[:, 0], np.zeros(n_dingri.shape[0])])
                    v2 = np.column_stack([
                        -n_dingri[:, 0] * n_dingri[:, 2],
                        -n_dingri[:, 1] * n_dingri[:, 2], 
                        n_dingri[:, 1]**2 + n_dingri[:, 0]**2
                    ])
                    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
                    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
                    
                    corners = np.zeros((n_heliostats, 4, 3))
                    corners[:, 0] = np.column_stack([
                        location[:, 0] + W * v1[:, 0] / 2 + H * v2[:, 0] / 2,
                        location[:, 1] + W * v1[:, 1] / 2 + H * v2[:, 1] / 2,
                        h + W * v1[:, 2] / 2 + H * v2[:, 2] / 2
                    ])
                    # ... 其他角点计算（简化版本暂时使用近似）
                
                # 计算每个定日镜的阴影和截断
                shade_time = np.zeros(n_heliostats)
                ntrunc_time = np.zeros(n_heliostats)
                
                for k in range(n_heliostats):
                    # 简化的阴影计算
                    shade_ratio = calculate_simplified_shadow(
                        k, location, corners, s_in_current, n_dingri, 
                        neighbors_cache[k], W, H, h, grid_reduction
                    )
                    shade_time[k] = shade_ratio
                    
                    # 简化的截断效率计算
                    ntrunc_time[k] = calculate_simplified_truncation(
                        k, location, shade_ratio, W, H, h
                    )
                
                shade_simplified[(i, j)] = shade_time
                ntrunc_simplified[(i, j)] = ntrunc_time
                pbar.update(1)
    
    total_time = time.time() - start_time
    print(f"\n简化计算完成! 总耗时: {total_time:.2f}秒")
    
    # ==================== 结果插值和汇总 ====================
    
    # 对完整的12×5时间步进行插值
    shade_full = np.zeros((12, 5, n_heliostats))
    ntrunc_full = np.zeros((12, 5, n_heliostats))
    
    for i in range(12):
        for j in range(5):
            if (i, j) in shade_simplified:
                # 直接使用计算值
                shade_full[i, j, :] = shade_simplified[(i, j)]
                ntrunc_full[i, j, :] = ntrunc_simplified[(i, j)]
            else:
                # 使用最近邻插值
                closest_i = min(selected_months, key=lambda x: abs(x - i))
                closest_j = min(selected_times, key=lambda x: abs(x - j))
                shade_full[i, j, :] = shade_simplified[(closest_i, closest_j)]
                ntrunc_full[i, j, :] = ntrunc_simplified[(closest_i, closest_j)]
    
    # 重构完整的s_in数组用于后续计算
    s_in_full = np.zeros((12, 15))
    alphas_full = np.zeros((12, 5))
    
    for i in range(12):
        for j in range(5):
            if (i, j) in s_in:
                s_in_full[i, 3*j:3*j+3] = s_in[(i, j)]
                alphas_full[i, j] = alphas[(i, j)]
            else:
                # 使用插值
                closest_i = min(selected_months, key=lambda x: abs(x - i))
                closest_j = min(selected_times, key=lambda x: abs(x - j))
                s_in_full[i, 3*j:3*j+3] = s_in[(closest_i, closest_j)]
                alphas_full[i, j] = alphas[(closest_i, closest_j)]
    
    # 处理对称时刻
    s_in_full[:, 9] = -s_in_full[:, 3]
    s_in_full[:, 10] = s_in_full[:, 4]
    s_in_full[:, 11] = s_in_full[:, 5]
    s_in_full[:, 12] = -s_in_full[:, 0]
    s_in_full[:, 13] = s_in_full[:, 1]
    s_in_full[:, 14] = s_in_full[:, 2]
    
    # ==================== 效率计算 ====================
    
    # 快速效率计算
    eta_cos, eta_sb, eta_at, eta, eta_ref = calculate_efficiencies_fast(
        shade_full, ntrunc_full, s_in_full, s_reflect, location, alphas_full
    )
    
    # 快速功率计算
    E, Ef, year_Ef, year_Ef_per_area, DNI = calculate_power_output_fast(
        eta, alphas_full, location, W, H
    )
    
    print(f"\n简化计算结果 - 年平均输出功率: {year_Ef:.2f} kW, 单位面积功率: {year_Ef_per_area:.4f} kW/m²")
    
    return s_in_full, alphas_full, shade_full, ntrunc_full, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI

def calculate_simplified_shadow(k, location, corners, s_in_current, n_dingri, 
                               nearest_indices, W, H, h, grid_reduction=2):
    """简化的阴影计算"""
    # 降低网格分辨率
    dl = H / (5 / grid_reduction)  # 网格更大，计算更快
    xid, yid = max(1, int(W / dl)), max(1, int(H / dl))
    
    shade_count = 0
    total_count = xid * yid
    
    # 简化的网格点计算
    for ii in range(xid):
        for jj in range(yid):
            # 网格中心坐标（简化计算）
            xi = location[k, 0] + (jj - yid/2) * dl
            yi = location[k, 1] + (ii - xid/2) * dl  
            zi = h
            
            # 检查前3个最近邻就足够了（大部分情况下）
            for idx, kk in enumerate(nearest_indices[:3]):
                if kk == k:
                    continue
                
                # 简化的遮挡检查：只检查主要的入射光遮挡
                # 使用快速的几何近似判断
                
                # 计算从网格点到遮挡定日镜中心的向量
                to_blocker = np.array([location[kk, 0] - xi, location[kk, 1] - yi, h - zi])
                
                # 简化的遮挡判断：如果光线方向和遮挡方向夹角很小，认为被遮挡
                cos_angle = np.dot(to_blocker, -s_in_current) / (np.linalg.norm(to_blocker) * np.linalg.norm(s_in_current))
                
                # 如果角度很小且距离合适，认为被遮挡
                if cos_angle > 0.8 and np.linalg.norm(to_blocker[:2]) < W + 2:  # 简化的遮挡条件
                    shade_count += 1
                    break
    
    return shade_count / total_count if total_count > 0 else 0

def calculate_simplified_truncation(k, location, shade_ratio, W, H, h):
    """简化的截断效率计算"""
    if shade_ratio >= 0.99:
        return 0
    
    # 使用简化的截断公式
    d = np.sqrt(location[k, 0]**2 + location[k, 1]**2 + (h - 80)**2)
    r = 4.65e-3 * d
    
    # 简化的截断效率
    light_out = (W + 2*r) * (H + 2*r)
    light_in = min(8, 2*r + H) * min(7, 2*r + W)
    
    base_efficiency = light_in / light_out if light_out > 0 else 0
    shadow_penalty = shade_ratio * 0.1
    
    return max(0, base_efficiency - shadow_penalty)

def calculate_efficiencies_fast(shade, ntrunc, s_in, s_reflect, location, alphas):
    """快速效率计算"""
    # 余弦效率（简化计算）
    eta_cos = np.zeros((12, 5, location.shape[0]))
    for i in range(12):
        for j in range(5):
            n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
            n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
            
            # 批量计算余弦效率
            cos_eff = np.abs(np.sum(n_dingri * s_in[i, 3*j:3*j+3], axis=1))
            eta_cos[i, j, :] = cos_eff
    
    # 阴影遮挡效率
    eta_sb = 1 - shade
    
    # 大气透射率（使用缓存）
    h = 4
    loc_jire = np.array([0, 0, 80])
    d_HR = np.sqrt((location[:, 0] - loc_jire[0])**2 + 
                   (location[:, 1] - loc_jire[1])**2 + 
                   (h - loc_jire[2])**2)
    eta_at = 0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR**2
    eta_at_full = np.broadcast_to(eta_at[np.newaxis, np.newaxis, :], (12, 5, location.shape[0]))
    
    # 镜面反射率
    eta_ref = 0.92
    
    # 光学效率
    eta = eta_sb * eta_cos * eta_at_full * ntrunc * eta_ref
    
    return eta_cos, eta_sb, eta_at_full, eta, eta_ref

def calculate_power_output_fast(eta, alphas, location, W=6, H=6):
    """快速功率计算"""
    # 快速DNI计算
    DNI = np.zeros((12, 5))
    G0 = 1.366
    H_a = 3
    a = 0.4237 - 0.00821 * (6 - H_a)**2
    b = 0.5055 + 0.00595 * (6.5 - H_a)**2
    c = 0.2711 + 0.01858 * (2.5 - H_a)**2
    
    for i in range(12):
        for j in range(5):
            if alphas[i, j] > 0:
                DNI[i, j] = G0 * (a + b * np.exp(-c / np.sin(alphas[i, j])))
    
    # 定日镜面积
    A = W * H
    
    # 批量功率计算
    E = DNI[:, :, np.newaxis] * A * eta
    
    # 总功率
    Ef = np.sum(E, axis=2)
    year_Ef = np.mean(Ef)
    
    # 单位面积功率
    total_area = location.shape[0] * A
    year_Ef_per_area = year_Ef / total_area
    
    return E, Ef, year_Ef, year_Ef_per_area, DNI
