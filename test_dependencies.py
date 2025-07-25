# 测试依赖库安装
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize_scalar
import pickle

def test_dependencies():
    """测试所有依赖库是否正常工作"""
    
    print("=" * 50)
    print("依赖库测试")
    print("=" * 50)
    
    # 测试NumPy
    try:
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✓ NumPy测试成功: 数组平均值 = {result}")
    except Exception as e:
        print(f"✗ NumPy测试失败: {e}")
    
    # 测试SciPy spatial
    try:
        points = np.array([[0, 0], [1, 1], [2, 2]])
        dist_matrix = distance.cdist(points, points)
        print(f"✓ SciPy spatial测试成功: 距离矩阵形状 = {dist_matrix.shape}")
    except Exception as e:
        print(f"✗ SciPy spatial测试失败: {e}")
    
    # 测试SciPy optimize
    try:
        def test_func(x):
            return (x - 2) ** 2
        result = minimize_scalar(test_func)
        print(f"✓ SciPy optimize测试成功: 最小值点 = {result.x:.2f}")
    except Exception as e:
        print(f"✗ SciPy optimize测试失败: {e}")
    
    # 测试Matplotlib
    try:
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("测试图形")
        plt.savefig("test_plot.png")
        plt.close()
        print("✓ Matplotlib测试成功: 测试图形已保存为 test_plot.png")
    except Exception as e:
        print(f"✗ Matplotlib测试失败: {e}")
    
    # 测试Pickle
    try:
        test_data = {"name": "测试", "values": [1, 2, 3]}
        with open("test_pickle.pkl", "wb") as f:
            pickle.dump(test_data, f)
        
        with open("test_pickle.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        
        print(f"✓ Pickle测试成功: 数据序列化和反序列化正常")
    except Exception as e:
        print(f"✗ Pickle测试失败: {e}")
    
    print("=" * 50)
    print("版本信息:")
    print(f"NumPy: {np.__version__}")
    import scipy
    print(f"SciPy: {scipy.__version__}")
    import matplotlib
    print(f"Matplotlib: {matplotlib.__version__}")
    print("=" * 50)
    print("所有依赖库测试完成！")

if __name__ == "__main__":
    test_dependencies()
