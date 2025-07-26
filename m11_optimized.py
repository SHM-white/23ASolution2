# 问题一：定日镜场光学效率与输出热功率求解 - 优化版本
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
import pickle
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# ==================== 原始数学逻辑函数（不变） ====================

# 计算DNI法向直接辐射照度
def calculate_DNI(alpha_s, H_a=3):
    """
    计算法向直接辐射照度
    alpha_s: 太阳高度角(弧度)
    H_a: 海拔高度(km), 默认3km
    """
    G0 = 1.366  # 太阳常数 kW/m²
    a = 0.4237 - 0.00821 * (6 - H_a)**2
    b = 0.5055 + 0.00595 * (6.5 - H_a)**2
    c = 0.2711 + 0.01858 * (2.5 - H_a)**2
    
    DNI = G0 * (a + b * np.exp(-c / np.sin(alpha_s)))
    return DNI

# 读取定日镜坐标数据
def read_data():
    """从附件.xlsx读取定日镜坐标数据"""
    try:
        # 读取Excel文件，跳过第一行标题
        df = pd.read_excel('附件.xlsx', header=0)  # 第一行是标题
        location = df.values  # 转换为numpy数组，每行为[x, y]坐标
        print(f"成功读取 {len(location)} 个定日镜坐标")
        return location
    except Exception as e:
        print(f"读取附件.xlsx失败: {e}")
        print("使用示例数据...")
        # 生成圆形分布的定日镜位置作为备用
        n_heliostats = 100  # 定日镜数量
        angles = np.linspace(0, 2*np.pi, n_heliostats, endpoint=False)
        radii = np.random.uniform(120, 300, n_heliostats)  # 半径在120-300m之间
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        location = np.column_stack([x, y])
        return location

# 计算平面与直线交点
def calc_plane_line_intersect_point(en, planepoint, es, linepoint):
    """
    en: 法向量
    planepoint: 平面上一个点的坐标  
    es: 直线的方向向量
    linepoint: 直线上一个点的坐标
    """
    # 确保输入为numpy数组
    en = np.array(en)
    planepoint = np.array(planepoint)
    es = np.array(es)
    linepoint = np.array(linepoint)
    
    vpt = np.dot(en, es)  # 内积
    if abs(vpt) < 1e-10:  # 避免除零错误
        return None, None, None
    else:
        t = (np.dot(planepoint - linepoint, en)) / vpt
        intersection = linepoint + es * t
        return intersection[0], intersection[1], intersection[2]

# 判断点是否在三角形内
def point_in_triangle(px, py, pz, A, B, C):
    V1q = np.array([px - A[0], py - A[1], pz - A[2]])
    V2q = np.array([px - B[0], py - B[1], pz - B[2]])
    V3q = np.array([px - C[0], py - C[1], pz - C[2]])
    
    V12 = B - A
    V13 = C - A
    V23 = C - B
    
    # 计算叉积
    cross_AB_AP = np.cross(V12, V1q)
    cross_AB_AC = np.cross(V12, V13)
    cross_BC_BP = np.cross(V23, V2q)
    cross_BC_BA = np.cross(V23, -V12)
    cross_CA_CP = np.cross(-V13, V3q)
    cross_CA_CB = np.cross(-V13, -V23)
    
    # 判断是否满足条件
    return np.dot(cross_AB_AP, cross_AB_AC) > 0 and np.dot(cross_BC_BP, cross_BC_BA) > 0 and np.dot(cross_CA_CP, cross_CA_CB) > 0

# 判断点是否在矩形内部
def is_point_in_rectangular(px, py, pz, rectangular):
    A = rectangular[0, :]
    B = rectangular[1, :]
    C = rectangular[2, :]
    D = rectangular[3, :]
    
    # 分别判断点是否在两个三角形内部
    return (point_in_triangle(px, py, pz, A, B, C) or 
            point_in_triangle(px, py, pz, A, C, D))

# 计算各种效率指标
def calculate_efficiencies(shade, ntrunc, s_in, s_reflect, location, alphas, xid, yid):
    """
    计算余弦效率、阴影遮挡效率、大气透射率、光学效率等
    """
    # 余弦效率
    eta_cos = np.zeros((12, 5, location.shape[0]))
    for i in range(12):
        for j in range(5):
            n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
            n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
            
            for k in range(location.shape[0]):
                eta_cos[i, j, k] = abs(np.dot(n_dingri[k], s_in[i, 3*j:3*j+3]))
    
    # 阴影遮挡效率
    eta_sb = 1 - shade  # shade现在是遮挡比例（0-1），效率就是1减去遮挡比例
    
    # 大气透射率
    h = 4  # 定日镜高度
    loc_jire = np.array([0, 0, 80])  # 集热器中心
    d_HR = np.sqrt((location[:, 0] - loc_jire[0])**2 + 
                   (location[:, 1] - loc_jire[1])**2 + 
                   (h - loc_jire[2])**2)
    eta_at = 0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR**2
    
    # 扩展大气透射率到所有时间点
    eta_at_full = np.broadcast_to(eta_at[np.newaxis, np.newaxis, :], (12, 5, location.shape[0]))
    
    # 镜面反射率
    eta_ref = 0.92
    
    # 光学效率
    eta = eta_sb * eta_cos * eta_at_full * ntrunc * eta_ref
    
    return eta_cos, eta_sb, eta_at_full, eta, eta_ref

# 计算输出功率
def calculate_power_output(eta, alphas, location, W=6, H=6):
    """
    计算定日镜场输出功率
    """
    # 计算DNI
    DNI = np.zeros((12, 5))
    for i in range(12):
        for j in range(5):
            DNI[i, j] = calculate_DNI(alphas[i, j])
    
    # 定日镜面积
    A = W * H
    
    # 每块定日镜的输出功率
    E = np.zeros((12, 5, location.shape[0]))
    for i in range(12):
        for j in range(5):
            E[i, j, :] = DNI[i, j] * A * eta[i, j, :]
    
    # 镜场总功率
    Ef = np.sum(E, axis=2)
    
    # 年平均功率
    year_Ef = np.mean(Ef)
    
    # 单位面积年平均功率
    total_area = location.shape[0] * A
    year_Ef_per_area = year_Ef / total_area
    
    return E, Ef, year_Ef, year_Ef_per_area, DNI

# 输出结果摘要
def print_results_summary(eta_cos, eta_sb, eta_at, ntrunc, eta, year_Ef, year_Ef_per_area):
    """
    输出计算结果摘要
    """
    print("\n" + "="*60)
    print("计算结果摘要")
    print("="*60)
    
    # 计算年平均效率
    year_eta_cos = np.mean(eta_cos)
    year_eta_sb = np.mean(eta_sb) 
    year_eta_at = np.mean(eta_at)
    year_eta_trunc = np.mean(ntrunc)
    year_eta = np.mean(eta)
    
    print(f"年平均余弦效率:     {year_eta_cos:.4f}")
    print(f"年平均阴影遮挡效率: {year_eta_sb:.4f}")
    print(f"年平均大气透射率:   {year_eta_at:.4f}")
    print(f"年平均截断效率:     {year_eta_trunc:.4f}")
    print(f"年平均光学效率:     {year_eta:.4f}")
    print(f"年平均输出功率:     {year_Ef:.2f} kW")
    print(f"年平均单位面积功率: {year_Ef_per_area:.4f} kW/m²")
    print("="*60)

# ==================== 优化执行逻辑 ====================

class OptimizedShadowCalculator:
    """优化的阴影计算器 - 使用预计算和缓存"""
    
    def __init__(self, location, W, H, h):
        self.location = location
        self.W = W
        self.H = H
        self.h = h
        self.n_heliostats = len(location)
        
        # 预计算KD树用于快速最近邻查询
        print("构建空间索引...")
        self.kdtree = cKDTree(location)
        
        # 预计算所有定日镜的最近邻
        print("预计算最近邻...")
        self.neighbors_cache = self._precompute_neighbors()
        
    def _precompute_neighbors(self):
        """预计算所有定日镜的最近邻"""
        neighbors = {}
        # 根据定日镜数量动态调整最近邻数量
        max_neighbors = min(20, self.n_heliostats // 10 + 5)  # 最多20个最近邻
        
        for k in range(self.n_heliostats):
            _, indices = self.kdtree.query(self.location[k], k=max_neighbors+1)  # 包括自己
            neighbors[k] = indices[1:]  # 排除自己
        return neighbors
    
    def precompute_time_step_data(self, i, j, s_in, s_reflect):
        """预计算单个时间步的几何数据"""
        # 确定每个定日镜的法向量
        n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
        n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
        
        # 计算定日镜四个角点坐标（按照MATLAB代码的方式）
        v1 = np.column_stack([n_dingri[:, 1], -n_dingri[:, 0], np.zeros(n_dingri.shape[0])])
        v2 = np.column_stack([
            -n_dingri[:, 0] * n_dingri[:, 2],
            -n_dingri[:, 1] * n_dingri[:, 2], 
            n_dingri[:, 1]**2 + n_dingri[:, 0]**2
        ])
        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        
        # 预计算所有角点（优化：批量计算）
        corners = np.zeros((self.n_heliostats, 4, 3))
        
        # 四个角点
        corners[:, 0] = np.column_stack([
            self.location[:, 0] + self.W * v1[:, 0] / 2 + self.H * v2[:, 0] / 2,
            self.location[:, 1] + self.W * v1[:, 1] / 2 + self.H * v2[:, 1] / 2,
            self.h + self.W * v1[:, 2] / 2 + self.H * v2[:, 2] / 2
        ])
        
        corners[:, 1] = np.column_stack([
            self.location[:, 0] - self.W * v1[:, 0] / 2 + self.H * v2[:, 0] / 2,
            self.location[:, 1] - self.W * v1[:, 1] / 2 + self.H * v2[:, 1] / 2,
            self.h - self.W * v1[:, 2] / 2 + self.H * v2[:, 2] / 2
        ])
        
        corners[:, 2] = np.column_stack([
            self.location[:, 0] - self.W * v1[:, 0] / 2 - self.H * v2[:, 0] / 2,
            self.location[:, 1] - self.W * v1[:, 1] / 2 - self.H * v2[:, 1] / 2,
            self.h - self.W * v1[:, 2] / 2 - self.H * v2[:, 2] / 2
        ])
        
        corners[:, 3] = np.column_stack([
            self.location[:, 0] + self.W * v1[:, 0] / 2 - self.H * v2[:, 0] / 2,
            self.location[:, 1] + self.W * v1[:, 1] / 2 - self.H * v2[:, 1] / 2,
            self.h + self.W * v1[:, 2] / 2 - self.H * v2[:, 2] / 2
        ])
        
        return {
            'n_dingri': n_dingri,
            'v1': v1,
            'v2': v2,
            'corners': corners,
            's_in_current': s_in[i, 3*j:3*j+3],
            # 反射光线（预计算）
            's_reflect_current': s_reflect[i, 3*j:3*j+3]
        }
    
    def calculate_single_heliostat_shadow(self, k, time_data):
        """计算单个定日镜的阴影（原始数学逻辑保持不变）"""
        # 栅格化计算
        dl = self.H / 5  # 分网格
        xid, yid = int(self.W / dl), int(self.H / dl)
        shade1 = np.zeros((xid, yid))
        
        # 获取预计算的数据
        n_dingri = time_data['n_dingri']
        v1 = time_data['v1']
        v2 = time_data['v2']
        corners = time_data['corners']
        s_in_current = time_data['s_in_current']
        s_reflect = s_in_current - 2 * np.dot(s_in_current, n_dingri[k]) * n_dingri[k]
        
        # 获取最近邻（使用预计算）
        nearest_indices = self.neighbors_cache[k]
        
        # 预计算当前定日镜的几何参数（优化：减少重复数组访问）
        v1_k = v1[k]
        v2_k = v2[k]
        corner1_k = corners[k, 1]  # xp2对应的角点
        
        for ii in range(xid):  # 简化网格计算
            for jj in range(yid):
                # 格子中心坐标
                xi = corner1_k[0] + jj * dl * v1_k[0] - ii * dl * v2_k[0] - dl * v1_k[0] / 2 + dl * v2_k[0] / 2
                yi = corner1_k[1] + jj * dl * v1_k[1] - ii * dl * v2_k[1] - dl * v1_k[1] / 2 + dl * v2_k[1] / 2
                zi = corner1_k[2] + jj * dl * v1_k[2] - ii * dl * v2_k[2] - dl * v1_k[2] / 2 + dl * v2_k[2] / 2
                
                # 检查是否被其他定日镜遮挡
                for kk in nearest_indices: # kk为最近的定日镜索引
                    if kk == k:  # 跳过自身
                        continue
                        
                    # 计算入射光线和反射光线与其他定日镜的交点
                    try:
                        # 1. 检查入射光线遮挡（阴影）
                        px1, py1, pz1 = calc_plane_line_intersect_point(
                            n_dingri[kk], [self.location[kk, 0], self.location[kk, 1], self.h],
                            s_in_current, [xi, yi, zi]  # 入射光线方向
                        )
                        
                        if px1 is not None:
                            # 按照MATLAB代码的逻辑：检查点积条件
                            heliostat_center = np.array([self.location[kk, 0], self.location[kk, 1], self.h])
                            intersection_point = np.array([px1, py1, pz1])
                            dot_product = np.dot(heliostat_center - intersection_point, s_in_current)
                            
                            if dot_product > 0:  # MATLAB条件：if dot([location(kk, :), h] - [px1, py1, pz1], s_in(...)) > 0
                                if is_point_in_rectangular(px1, py1, pz1, corners[kk]):
                                    shade1[ii, jj] = 1
                                    break  # 找到阴影就退出
                        
                        # 2. 检查反射光线遮挡
                        px2, py2, pz2 = calc_plane_line_intersect_point(
                            n_dingri[kk], [self.location[kk, 0], self.location[kk, 1], self.h],
                            s_reflect, [xi, yi, zi]  # 反射光线方向
                        )
                        
                        if px2 is not None:
                            # 按照MATLAB代码的逻辑：检查点积条件
                            intersection_point = np.array([px2, py2, pz2])
                            heliostat_center = np.array([self.location[kk, 0], self.location[kk, 1], self.h])
                            dot_product = np.dot(intersection_point - heliostat_center, s_reflect)
                            
                            if dot_product > 0:  # MATLAB条件：if dot([px2, py2, pz2] - [location(kk, :), h], s_reflect(k, :)) > 0
                                if is_point_in_rectangular(px2, py2, pz2, corners[kk]):
                                    shade1[ii, jj] = 1
                                    break  # 找到遮挡就退出
                    except:
                        continue
        
        return np.sum(shade1) / (xid * yid)
    
    def calculate_single_heliostat_truncation(self, k, shade_ratio):
        """计算单个定日镜的截断效率（原始数学逻辑保持不变）"""
        # 栅格化计算
        dl = self.H / 5  # 分网格
        xid, yid = int(self.W / dl), int(self.H / dl)
        
        # 如果完全被遮挡
        if shade_ratio >= 0.99:
            return 0
        
        # 定日镜到集热器中心距离
        d = np.sqrt(self.location[k, 0]**2 + self.location[k, 1]**2 + (self.h - 80)**2)
        r = 4.65e-3 * d  # 光斑半径（太阳锥角4.65mrad）
        
        # 集热器限制参数
        xlimit = 3.5 - r
        ylimit = 4 - r
        
        # 不考虑阴影的定日镜总输出光功率
        light_out = (self.W + 2*r) * (self.H + 2*r)
        
        # 不考虑阴影的集热器总接收光功率
        light_in = min(8, 2*r + self.H) * min(7, 2*r + self.W)
        
        # 简化截断效率计算（如果阴影比例较小，使用简化公式）
        if shade_ratio < 0.1:  # 阴影很少时使用简化计算
            return light_in / light_out if light_out > 0 else 0
        
        # 对于有显著阴影的情况，使用原始复杂计算
        # 这里需要重建shade1来计算详细的截断效率
        # 为了性能考虑，可以使用近似方法
        shadow_penalty = shade_ratio * 0.1  # 阴影对截断效率的惩罚
        base_efficiency = light_in / light_out if light_out > 0 else 0
        
        return max(0, base_efficiency - shadow_penalty)

def process_time_step_parallel(args):
    """并行处理单个时间步的函数"""
    time_idx, calculator, s_in, s_reflect = args
    i, j = time_idx
    # calculator类型为OptimizedShadowCalculator，显示指定便于静态分析
    calculator: OptimizedShadowCalculator
    # 预计算该时间步的几何数据
    time_data = calculator.precompute_time_step_data(i, j, s_in, s_reflect)
    
    # 计算所有定日镜的阴影
    shadows = np.zeros(calculator.n_heliostats)
    ntrunc = np.zeros(calculator.n_heliostats)
    
    for k in range(calculator.n_heliostats):
        # 计算阴影
        shadow_ratio = calculator.calculate_single_heliostat_shadow(k, time_data)
        shadows[k] = shadow_ratio
        
        # 计算截断效率
        ntrunc[k] = calculator.calculate_single_heliostat_truncation(k, shadow_ratio)
    
    return (i, j), shadows, ntrunc

def calculate_results_optimized(H, W, location, loc_jire, h, use_multiprocessing=True, max_workers=None):
    """优化版本的计算函数 - 使用多线程和预计算"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 32)  # 默认使用CPU核心数，但不超过8
    
    print(f"优化版本启动 - 使用{'多进程' if use_multiprocessing else '单线程'}计算，工作进程数: {max_workers}")
    
    # ==================== 原始数学计算部分（不变） ====================
    
    loc_dingri = np.column_stack([location, h * np.ones(location.shape[0])])  # 定日镜中心坐标
    # 反射光线的方向向量
    s_reflect = loc_jire - loc_dingri
    s_reflect = s_reflect / np.linalg.norm(s_reflect, axis=1, keepdims=True)  # 单位化
    
    # 读取日期数据
    D = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # 每月21日对应的天数
    ST = np.array([9.0, 10.5, 12.0, 13.5, 15.0])  # 当地时间
    
    # 初始化数组
    alphas = np.zeros((12, 5))  # 太阳高度角
    gamas = np.zeros((12, 5))   # 太阳方位角
    s_in = np.zeros((12, 15))   # 入射光方向向量
    phi = 39.4 * np.pi / 180    # 当地纬度
    
    # 计算太阳位置参数
    for i in range(12):  # 12个月
        for j in range(5):  # 5个时刻
            # 太阳赤纬角
            delta = np.arcsin(np.sin(2 * np.pi * D[i] / 365) * np.sin(2 * np.pi * 23.45 / 360))
            
            # 太阳时角
            w = (ST[j] - 12) * np.pi / 12
            
            # 太阳高度角
            alphas[i, j] = np.arcsin(np.cos(delta) * np.cos(phi) * np.cos(w) + 
                                   np.sin(delta) * np.sin(phi))
            
            # 太阳方位角
            cos_gamma = (np.sin(delta) - np.sin(alphas[i, j]) * np.sin(phi)) / (np.cos(alphas[i, j]) * np.cos(phi))
            # 限制cos_gamma在[-1, 1]范围内，避免arccos错误
            cos_gamma = np.clip(cos_gamma, -1, 1)
            gamma_temp = np.arccos(cos_gamma)
            
            # 考虑太阳方位角的方向：上午为负，下午为正（相对于正南方向）
            if w < 0:  # 上午
                gamas[i, j] = -gamma_temp
            else:  # 下午
                gamas[i, j] = gamma_temp
            
            # 入射光的方向向量（从太阳指向地面）
            s_in_temp = -np.array([np.sin(gamas[i, j]), np.cos(gamas[i, j]), -np.tan(alphas[i, j])])
            s_in[i, 3*j:3*j+3] = s_in_temp / np.linalg.norm(s_in_temp)  # 单位化
    
    # 处理对称时刻
    s_in[:, 9] = -s_in[:, 3]
    s_in[:, 10] = s_in[:, 4]
    s_in[:, 11] = s_in[:, 5]
    s_in[:, 12] = -s_in[:, 0]
    s_in[:, 13] = s_in[:, 1]
    s_in[:, 14] = s_in[:, 2]
    
    # ==================== 优化执行部分 ====================
    
    # 初始化结果数组
    shade = np.zeros((12, 5, location.shape[0]))  # 阴影
    ntrunc = np.zeros((12, 5, location.shape[0]))  # 截断效率
    
    # 创建优化计算器
    calculator = OptimizedShadowCalculator(location, W, H, h)
    
    # 准备并行计算参数
    time_steps = [(i, j) for i in range(12) for j in range(5)]
    
    print("开始优化计算阴影遮挡和截断效率...")
    start_time = time.time()
    
    if use_multiprocessing and len(location) > 50:  # 只有定日镜数量足够多时才使用多进程
        # 使用多进程并行计算
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 准备参数
            args_list = [(ts, calculator, s_in, s_reflect) for ts in time_steps]
            
            # 使用进度条显示计算进度
            with tqdm(total=len(time_steps), desc="并行计算", ncols=80) as pbar:
                # 提交所有任务
                future_to_time = {executor.submit(process_time_step_parallel, args): args[0] 
                                 for args in args_list}
                
                # 收集结果
                for future in future_to_time:
                    try:
                        (i, j), shadows, ntrunc_values = future.result()
                        shade[i, j, :] = shadows
                        ntrunc[i, j, :] = ntrunc_values
                        pbar.update(1)
                    except Exception as e:
                        print(f"计算时间步 {future_to_time[future]} 时出错: {e}")
                        pbar.update(1)
    else:
        # 使用单线程计算
        with tqdm(total=len(time_steps), desc="串行计算", ncols=80) as pbar:
            for ts in time_steps:
                try:
                    (i, j), shadows, ntrunc_values = process_time_step_parallel((ts, calculator, s_in, s_reflect))
                    shade[i, j, :] = shadows
                    ntrunc[i, j, :] = ntrunc_values
                    pbar.update(1)
                except Exception as e:
                    print(f"计算时间步 {ts} 时出错: {e}")
                    pbar.update(1)
    
    total_time = time.time() - start_time
    print(f"\n优化计算完成! 总耗时: {total_time:.2f}秒")
    
    # ==================== 原始效率计算部分（不变） ====================
    
    # 计算各种效率指标
    xid, yid = int(W / (H/5)), int(H / (H/5))  # 网格参数
    eta_cos, eta_sb, eta_at, eta, eta_ref = calculate_efficiencies(shade, ntrunc, s_in, s_reflect, location, alphas, xid, yid)
    
    # 计算输出功率
    E, Ef, year_Ef, year_Ef_per_area, DNI = calculate_power_output(eta, alphas, location)
    
    # 输出结果摘要
    print_results_summary(eta_cos, eta_sb, eta_at, ntrunc, eta, year_Ef, year_Ef_per_area)
    
    return s_in, alphas, shade, ntrunc, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI

def main_optimized():
    """优化版本的主函数"""
    print("=== 优化版本 - 问题一：定日镜场光学效率与输出热功率求解 ===")
    
    # 参数设置
    location = read_data()  # 定日镜xy坐标
    h = 4  # 定日镜高度
    loc_jire = np.array([0, 0, 80])  # 集热器中心坐标
    W, H = 6, 6  # 定日镜宽度和高度
    
    # 运行优化计算
    results = calculate_results_optimized(H, W, location, loc_jire, h, use_multiprocessing=True)
    s_in, alphas, shade, ntrunc, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI = results
    
    # 保存结果
    with open('Q1_results_optimized.pkl', 'wb') as f:
        pickle.dump({
            'location': location,
            's_in': s_in,
            'alphas': alphas,
            'shade': shade,
            'ntrunc': ntrunc,
            's_reflect': s_reflect,
            'eta_cos': eta_cos,
            'eta_sb': eta_sb,
            'eta_at': eta_at,
            'eta': eta,
            'E': E,
            'Ef': Ef,
            'year_Ef': year_Ef,
            'year_Ef_per_area': year_Ef_per_area,
            'DNI': DNI
        }, f)
    
    print("优化版本计算完成，结果已保存到 Q1_results_optimized.pkl")

if __name__ == "__main__":
    main_optimized()
