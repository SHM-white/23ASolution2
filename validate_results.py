import pickle
import numpy as np

# 加载优化版本结果
with open('Q1_results_optimized.pkl', 'rb') as f:
    opt_results = pickle.load(f)

print("=== 优化版本结果验证 ===")
print(f"定日镜数量: {len(opt_results['location'])}")
print(f"年平均光学效率: {np.mean(opt_results['eta']):.4f}")
print(f"年平均输出功率: {opt_results['year_Ef']:.2f} kW")
print(f"年平均单位面积功率: {opt_results['year_Ef_per_area']:.4f} kW/m²")

print("\n=== 效率分解 ===")
print(f"余弦效率: {np.mean(opt_results['eta_cos']):.4f}")
print(f"阴影遮挡效率: {np.mean(opt_results['eta_sb']):.4f}")
print(f"大气透射率: {np.mean(opt_results['eta_at']):.4f}")
print(f"截断效率: {np.mean(opt_results['ntrunc']):.4f}")

print("\n=== 数据统计 ===")
print(f"阴影比例范围: {np.min(1-opt_results['eta_sb']):.4f} - {np.max(1-opt_results['eta_sb']):.4f}")
print(f"截断效率范围: {np.min(opt_results['ntrunc']):.4f} - {np.max(opt_results['ntrunc']):.4f}")
print(f"光学效率范围: {np.min(opt_results['eta']):.4f} - {np.max(opt_results['eta']):.4f}")

# 检查是否有异常值
shade_ratio = 1 - opt_results['eta_sb']
print(f"\n=== 异常值检测 ===")
print(f"完全被遮挡的网格点数: {np.sum(shade_ratio >= 1.0)}")
print(f"无遮挡的网格点数: {np.sum(shade_ratio == 0.0)}")
print(f"有部分遮挡的网格点数: {np.sum((shade_ratio > 0.0) & (shade_ratio < 1.0))}")

# 显示月份和时刻的变化
print(f"\n=== 时间变化分析 ===")
monthly_avg = np.mean(opt_results['eta'], axis=(1, 2))
hourly_avg = np.mean(opt_results['eta'], axis=(0, 2))

print("月平均光学效率:")
months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
for i, (month, eff) in enumerate(zip(months, monthly_avg)):
    print(f"  {month}: {eff:.4f}")

print("时刻平均光学效率:")
times = ['9:00', '10:30', '12:00', '13:30', '15:00']
for i, (time, eff) in enumerate(zip(times, hourly_avg)):
    print(f"  {time}: {eff:.4f}")
