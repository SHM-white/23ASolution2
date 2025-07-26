"""
快速测试简化计算版本的效果
"""
import time
import numpy as np
from m21 import generate_heliostats, objfun2

def test_simplified_optimization():
    """测试简化版本的优化效果"""
    print("="*60)
    print("问题2简化计算测试")
    print("="*60)
    
    # 测试不同D值的性能
    test_D_values = [0, 50, 100, 150, 200, 250]
    results = []
    
    print("测试不同D值的功率密度:")
    print("D值  | 功率密度(kW/m²) | 定日镜数量 | 计算耗时(秒)")
    print("-" * 55)
    
    for D in test_D_values:
        start_time = time.time()
        location = generate_heliostats(D, 8)
        
        try:
            _, power_per_area = objfun2(8, 8, 6, D, location, simple=False)
            calc_time = time.time() - start_time
            results.append((D, power_per_area, len(location), calc_time))
            print(f"{D:3d}  |    {power_per_area:.4f}    |   {len(location):4d}    |   {calc_time:.2f}")
        except Exception as e:
            print(f"{D:3d}  |     ERROR      |   ----    |   ----")
            print(f"      错误: {e}")
    
    if results:
        # 找到最优D值
        best_D, best_power, best_count, best_time = max(results, key=lambda x: x[1])
        print("-" * 55)
        print(f"最优结果: D={best_D}, 功率密度={best_power:.4f} kW/m²")
        print(f"定日镜数量: {best_count}, 单次计算耗时: {best_time:.2f}秒")
        
        # 估算完整优化时间
        total_time = sum(r[3] for r in results)
        estimated_full_optimization = total_time * 10  # 假设三分搜索需要10倍时间
        print(f"总测试时间: {total_time:.1f}秒")
        print(f"估算完整三分搜索时间: {estimated_full_optimization:.1f}秒 ({estimated_full_optimization/60:.1f}分钟)")
    
    return results

if __name__ == "__main__":
    test_simplified_optimization()
