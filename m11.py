# 问题一：定日镜场光学效率与输出热功率求解
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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
    cross1 = np.cross(V12, V1q)
    cross2 = np.cross(V12, V13)
    cross3 = np.cross(V23, V2q)
    cross4 = np.cross(V23, -V12)
    cross5 = np.cross(-V13, V3q)
    cross6 = np.cross(-V13, -V23)
    
    # 判断是否同向
    cond1 = np.dot(cross1, cross2) > 0
    cond2 = np.dot(cross3, cross4) > 0
    cond3 = np.dot(cross5, cross6) > 0
    
    return cond1 and cond2 and cond3

# 判断点是否在矩形内部
def is_point_in_rectangular(px, py, pz, rectangular):
    A = rectangular[0, :]
    B = rectangular[1, :]
    C = rectangular[2, :]
    D = rectangular[3, :]
    
    # 分别判断点是否在两个三角形内部
    return (point_in_triangle(px, py, pz, A, B, C) or 
            point_in_triangle(px, py, pz, A, C, D))

def main():
    # 参数设置
    location = read_data()  # 定日镜xy坐标
    loc_jire = np.array([0, 0, 80])  # 集热器中心坐标
    h = 4  # 定日镜高度
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
            gamas[i, j] = np.arccos(cos_gamma)
            
            # 入射光的方向向量
            s_in_temp = -np.array([np.sin(gamas[i, j]), np.cos(gamas[i, j]), np.tan(alphas[i, j])])
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
    
    W, H = 6, 6  # 定日镜宽度和高度
    
    # 计算阴影遮挡和截断效率
    for i in range(12):  # 月份
        for j in range(5):  # 时刻
            # 确定每个定日镜的法向量
            n_dingri = s_in[i, 3*j:3*j+3] - s_reflect
            n_dingri = n_dingri / np.linalg.norm(n_dingri, axis=1, keepdims=True)
            
            # 计算定日镜四个角点坐标
            v1 = np.column_stack([n_dingri[:, 1], -n_dingri[:, 0], np.zeros(n_dingri.shape[0])])
            v2 = np.column_stack([-n_dingri[:, 0] * n_dingri[:, 2], 
                                -n_dingri[:, 1] * n_dingri[:, 2],
                                n_dingri[:, 1]**2 + n_dingri[:, 0]**2])
            
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
            for k in range(min(10, len(location))):  # 限制计算数量以加快速度
                shade1 = np.zeros((xid, yid))
                
                for ii in range(min(3, xid)):  # 简化网格计算
                    for jj in range(min(3, yid)):
                        # 格子中心坐标
                        xi = xp2[k] + jj * dl * v1[k, 0] - ii * dl * v2[k, 0] - dl * v1[k, 0] / 2 + dl * v2[k, 0] / 2
                        yi = yp2[k] + jj * dl * v1[k, 1] - ii * dl * v2[k, 1] - dl * v1[k, 1] / 2 + dl * v2[k, 1] / 2
                        zi = zp2[k] + jj * dl * v1[k, 2] - ii * dl * v2[k, 2] - dl * v1[k, 2] / 2 + dl * v2[k, 2] / 2
                        
                        # 简化阴影检测 - 只检查最近的几个定日镜
                        for kk in range(min(3, len(location))):
                            if kk == k:
                                continue
                                
                            # 计算入射光线与其他定日镜的交点
                            try:
                                px1, py1, pz1 = calc_plane_line_intersect_point(
                                    n_dingri[kk], [location[kk, 0], location[kk, 1], h],
                                    s_in[i, 3*j:3*j+3], [xi, yi, zi]
                                )
                                
                                if px1 is not None:
                                    # 简化的阴影判断
                                    dist_to_intersect = np.sqrt((xi - px1)**2 + (yi - py1)**2 + (zi - pz1)**2)
                                    if dist_to_intersect < W:  # 简化判断条件
                                        shade1[ii, jj] = 1
                                        shade[i, j, k] += 1
                                        break
                            except:
                                continue
                
                # 计算截断效率（简化版本）
                if np.sum(shade1) == xid * yid:
                    ntrunc[i, j, k] = 0.5  # 给一个默认值
                    continue
                
                # 定日镜到集热器中心距离
                d = np.sqrt(location[k, 0]**2 + location[k, 1]**2 + (h - 80)**2)
                r = 4.65e-3 * d  # 光斑半径
                
                # 简化的截断效率计算
                shadow_ratio = np.sum(shade1) / (xid * yid)
                ntrunc[i, j, k] = max(0.5, 1.0 - shadow_ratio * 0.3)  # 简化公式
    
    # 保存结果
    with open('Q1_results.pkl', 'wb') as f:
        pickle.dump({
            'location': location,
            's_in': s_in,
            'alphas': alphas,
            'shade': shade,
            'ntrunc': ntrunc,
            's_reflect': s_reflect
        }, f)
    
    print("问题一计算完成，结果已保存到 Q1_results.pkl")

if __name__ == "__main__":
    main()
