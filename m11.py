# 问题一：定日镜场光学效率与输出热功率求解
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
import pandas as pd
from tqdm import tqdm
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['黑体', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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

# 获取定日镜四个角点坐标
def get_heliostat_corners(location, h):
    corners = []
    for loc in location:
        x, y = loc
        corners.append(np.array([[x - 3, y - 3, h],
                                  [x + 3, y - 3, h],
                                  [x + 3, y + 3, h],
                                  [x - 3, y + 3, h]]))
    return np.array(corners)

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

def calculate_results(H, W, location, loc_jire, h):
    """计算在输入定日镜和集热器布局下的光学效率等结果

    Args:
        H (float): 定日镜高度
        W (float): 定日镜宽度
        location (np.array): 定日镜位置坐标
        loc_jire (np.array): 集热器位置坐标
        h (float): 定日镜高度
    """

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
    
    # 初始化阴影和截断效率
    shade = np.zeros((12, 5, location.shape[0]))  # 阴影
    ntrunc = np.zeros((12, 5, location.shape[0]))  # 截断效率
    
    # 性能优化：预计算所有定日镜的最近邻（避免重复计算）
    print("预计算最近邻以加速计算...")
    from scipy.spatial import cKDTree
    try:
        kdtree = cKDTree(location)
        neighbors_cache = {}
        for k in range(location.shape[0]):
            _, indices = kdtree.query(location[k], k=6)  # 包括自己
            neighbors_cache[k] = indices[1:]  # 排除自己，保留5个最近邻
        print("最近邻预计算完成")
    except ImportError:
        print("未安装scipy，使用原始方法")
        neighbors_cache = None
    
    
    
    # 计算阴影遮挡和截断效率
    print("开始计算阴影遮挡和截断效率...")
    total_calculations = 12 * 5 * location.shape[0]  # 总计算量
    
    # 使用tqdm创建进度条，显示速度、剩余时间等信息
    with tqdm(total=total_calculations, 
              desc="计算进度", 
              ncols=60,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        start_time = time.time()
        for i in range(12):  # 月份
            for j in range(5):  # 时刻
                # 更新进度条描述，显示当前处理的月份和时刻
                pbar.set_description(f"{i}-{j}")
                
                # 确定每个定日镜的法向量
                n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
                n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
            
                # 计算定日镜四个角点坐标
                
                v1 = np.column_stack([n_dingri[:, 1], -n_dingri[:, 0], np.zeros(n_dingri.shape[0])])
                
                '''v2 = np.column_stack([-n_dingri[:, 0] * n_dingri[:, 2], 
                                    -n_dingri[:, 1] * n_dingri[:, 2],
                                    n_dingri[:, 1]**2 + n_dingri[:, 0]**2])
                '''
                v2 = np.cross(n_dingri, v1)  # 计算垂直于法向量的第二个方向
                v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
                v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
                
                # 四个角点
                xp1 = location[:, 0] + W * v1[:, 0] / 2 + H * v2[:, 0] / 2
                yp1 = location[:, 1] + W * v1[:, 1] / 2 + H * v2[:, 1] / 2
                zp1 = h + W * v1[:, 2] / 2 + H * v2[:, 2] / 2
                
                xp2 = location[:, 0] - W * v1[:, 0] / 2 + H * v2[:, 0] / 2
                yp2 = location[:, 1] - W * v1[:, 1] / 2 + H * v2[:, 1] / 2
                zp2 = h - W * v1[:, 2] / 2 + H * v2[:, 2] / 2
                
                xp3 = location[:, 0] - W * v1[:, 0] / 2 - H * v2[:, 0] / 2
                yp3 = location[:, 1] - W * v1[:, 1] / 2 - H * v2[:, 1] / 2
                zp3 = h - W * v1[:, 2] / 2 - H * v2[:, 2] / 2
                
                xp4 = location[:, 0] + W * v1[:, 0] / 2 - H * v2[:, 0] / 2
                yp4 = location[:, 1] + W * v1[:, 1] / 2 - H * v2[:, 1] / 2
                zp4 = h + W * v1[:, 2] / 2 - H * v2[:, 2] / 2
                
                # 栅格化计算
                dl = H / 5  # 分网格
                xid, yid = int(W / dl), int(H / dl)
                
                # 阴影遮挡计算
                for k in range(len(location)):  
                    # 计算进度
                    # print(f"计算 {i + 1} 月第 {j + 1}/5 时刻定日镜 {k+1}/{len(location)} 的阴影遮挡...", end="", flush=True)
                    # 计算遮挡效率
                    shade1 = np.zeros((xid, yid))
                    
                    # 计算当前定日镜k最近的几个定日镜（优化：使用预计算的邻居）
                    if neighbors_cache is not None:
                        nearest_indices = neighbors_cache[k]
                    else:
                        distances = np.linalg.norm(location - location[k], axis=1)
                        nearest_indices = np.argsort(distances)[1:6]  
                    
                    for ii in range(xid):  # 简化网格计算
                        for jj in range(yid):
                            # 格子中心坐标
                            xi = xp2[k] + jj * dl * v1[k, 0] - ii * dl * v2[k, 0] - dl * v1[k, 0] / 2 + dl * v2[k, 0] / 2
                            yi = yp2[k] + jj * dl * v1[k, 1] - ii * dl * v2[k, 1] - dl * v1[k, 1] / 2 + dl * v2[k, 1] / 2
                            zi = zp2[k] + jj * dl * v1[k, 2] - ii * dl * v2[k, 2] - dl * v1[k, 2] / 2 + dl * v2[k, 2] / 2
                            
                            # 检查是否被其他定日镜遮挡入射光线和反射光线
                            for kk in nearest_indices: # kk为最近的定日镜索引
                                # 计算入射光线和反射光线与其他定日镜的交点（优先入射光线）
                                try:
                                    # 计算从网格点沿入射光线方向与kk定日镜平面的交点
                                    px1, py1, pz1 = calc_plane_line_intersect_point(
                                        n_dingri[kk], [location[kk, 0], location[kk, 1], h],
                                        -s_in[i, 3*j:3*j+3], [xi, yi, zi]  # 入射光线方向（负号表示从太阳到网格点）
                                    )
                                    # 检查交点是否在kk定日镜的矩形区域内
                                    if px1 is not None and is_point_in_rectangular(px1, py1, pz1,
                                                                np.array([[xp1[kk], yp1[kk], zp1[kk]],
                                                                        [xp2[kk], yp2[kk], zp2[kk]],
                                                                        [xp3[kk], yp3[kk], zp3[kk]],
                                                                        [xp4[kk], yp4[kk], zp4[kk]]])
                                        ) and pz1 > zi:  # 遮挡定日镜在当前网格点上方
                                        shade1[ii, jj] = 1
                                        break  # 一旦发现遮挡就跳出
                                    # 计算从网格点沿反射光线方向与kk定日镜平面的交点
                                    px2, py2, pz2 = calc_plane_line_intersect_point(
                                        n_dingri[kk], [location[kk, 0], location[kk, 1], h],
                                        s_reflect[kk], [xi, yi, zi]  # 反射光线方向
                                    )
                                    # 检查交点是否在kk定日镜的矩形区域内
                                    if px2 is not None and is_point_in_rectangular(px2, py2, pz2,
                                                                np.array([[xp1[kk], yp1[kk], zp1[kk]],
                                                                        [xp2[kk], yp2[kk], zp2[kk]],
                                                                        [xp3[kk], yp3[kk], zp3[kk]],
                                                                        [xp4[kk], yp4[kk], zp4[kk]]])
                                        ) and pz2 > zi:  # 遮挡定日镜在当前网格点上方
                                        shade1[ii, jj] = 1
                                        break  # 一旦发现遮挡就跳出
                                except:
                                    continue
                                
                    
                    # 累计阴影效果（修正：应该按网格比例计算，而不是简单累加）
                    shade[i, j, k] = np.sum(shade1) / (xid * yid)
                    
                    # 计算截断效率
                    if np.sum(shade1) == xid * yid:
                        ntrunc[i, j, k] = np.inf
                        continue
                    
                    # 定日镜到集热器中心距离
                    d = np.sqrt(location[k, 0]**2 + location[k, 1]**2 + (h - 80)**2)
                    r = 4.65e-3 * d  # 光斑半径（太阳锥角4.65mrad）
                    
                    # 集热器限制参数
                    xlimit = 3.5 - r
                    ylimit = 4 - r
                    
                    # 不考虑阴影的定日镜总输出光功率
                    light_out = (W + 2*r) * (H + 2*r)
                    
                    # 不考虑阴影的集热器总接收光功率
                    light_in = min(8, 2*r + H) * min(7, 2*r + W)
                    
                    # 处理有阴影的栅格
                    xx, yy = np.where(shade1 > 0)  # 阴影栅格索引
                    
                    for ix in range(len(xx)):
                        # 栅格在定日镜局部坐标系中的位置
                        yi_local = H/2 - xx[ix]*dl + dl/2
                        xi_local = -W/2 + yy[ix]*dl - dl/2
                        
                        # 判断栅格位置类型并计算相应的光功率损失
                        is_corner = ((xx[ix] == 0 and yy[ix] == 0) or 
                                (xx[ix] == 0 and yy[ix] == xid-1) or 
                                (xx[ix] == yid-1 and yy[ix] == 0) or 
                                (xx[ix] == yid-1 and yy[ix] == xid-1))
                        
                        is_left_right_edge = ((xx[ix] == 0 and 0 < yy[ix] < xid-1) or 
                                            (xx[ix] == yid-1 and 0 < yy[ix] < xid-1))
                        
                        is_top_bottom_edge = ((yy[ix] == 0 and 0 < xx[ix] < yid-1) or 
                                            (yy[ix] == xid-1 and 0 < xx[ix] < yid-1))
                        
                        if is_corner:  # 四个角
                            if r + abs(xi_local) + dl/2 < 3.5:  # 集热器完全吸收
                                light_out -= (dl + r)**2
                                light_in -= (dl + r)**2
                            elif r + abs(yi_local) + dl/2 < 4:  # 上下完全吸收，左右溢出
                                light_out -= (dl + r)**2
                                light_in -= (dl/2 + 3.5 - abs(xi_local)) * (dl + r)
                            else:  # 上下左右都溢出
                                light_out -= (dl + r)**2
                                light_in -= (dl/2 + 3.5 - abs(xi_local)) * (dl/2 + 4 - abs(yi_local))
                        
                        elif is_left_right_edge:  # 左右边缘
                            if r + abs(yi_local) + dl/2 < 4:  # 集热器完全吸收
                                light_out -= (dl + r) * dl
                                light_in -= (dl + r) * dl
                            else:  # 上下溢出
                                light_out -= (dl + r) * dl
                                light_in -= (dl/2 + 4 - abs(yi_local)) * dl
                        
                        elif is_top_bottom_edge:  # 上下边缘
                            if r + abs(xi_local) + dl/2 < 3.5:  # 集热器完全吸收
                                light_out -= (dl + r) * dl
                                light_in -= (dl + r) * dl
                            else:  # 左右溢出
                                light_out -= (dl + r) * dl
                                light_in -= (dl/2 + 3.5 - abs(xi_local)) * dl
                        
                        else:  # 中心区域
                            light_out -= dl**2
                            light_in -= dl**2
                    
                    # 计算截断效率
                    ntrunc[i, j, k] = light_in / light_out if light_out > 0 else np.inf
                    
                    # 更新进度条
                    pbar.update(1)
        
        # 计算完成，显示统计信息
        total_time = time.time() - start_time
        pbar.set_description(f"计算完成! 耗时: {total_time:.1f}秒")
    
    
    
    # 计算各种效率指标
    eta_cos, eta_sb, eta_at, eta, eta_ref = calculate_efficiencies(shade, ntrunc, s_in, s_reflect, location, alphas, xid, yid)
    
    # 计算输出功率
    E, Ef, year_Ef, year_Ef_per_area, DNI = calculate_power_output(eta, alphas, location)
    
    # 输出结果摘要
    print_results_summary(eta_cos, eta_sb, eta_at, ntrunc, eta, year_Ef, year_Ef_per_area)
    return s_in, alphas, shade, ntrunc, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI

def main():
    # 参数设置
    location = read_data()  # 定日镜xy坐标
    h = 4  # 定日镜高度
    loc_jire = np.array([0, 0, 80])  # 集热器中心坐标
    W, H = 6, 6  # 定日镜宽度和高度
    s_in, alphas, shade, ntrunc, s_reflect, eta_cos, eta_sb, eta_at, eta, E, Ef, year_Ef, year_Ef_per_area, DNI = calculate_results(H, W, location, loc_jire, h)

    # 保存结果
    with open('Q1_results.pkl', 'wb') as f:
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
    
    print("问题一计算完成，结果已保存到 Q1_results.pkl")

if __name__ == "__main__":
    main()
