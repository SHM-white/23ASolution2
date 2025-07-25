# 定日镜场优化设计项目运行指南

## 环境信息
- Python版本: 3.12.10
- 虚拟环境路径: `C:/Users/11312/Documents/source/py/23A/.venv/`
- Python执行命令: `C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe`

## 已安装的依赖库
- **NumPy 2.3.2** - 数值计算库
- **SciPy 1.16.0** - 科学计算库（包含spatial和optimize模块）
- **Matplotlib 3.10.3** - 绘图库
- **pickle** - Python内置序列化库

## 项目文件说明

### 主要Python文件
1. **m11.py** - 问题一：定日镜场光学效率与输出热功率求解
   - 实现太阳位置计算
   - 阴影遮挡和截断效率计算
   - 栅格化模拟算法

2. **m12.py** - 问题一：计算各种效率指标
   - 余弦效率计算
   - 阴影遮挡效率计算
   - 大气透射率计算
   - 结果可视化

3. **m21.py** - 问题二：三分查找优化吸收塔位置
   - 吸收塔位置优化
   - 定日镜宽度优化
   - 三分查找算法实现

4. **m22.py** - 问题二：验证定日镜尺寸和间距关系
   - W-H关系验证
   - W-dr关系验证
   - 参数优化验证

5. **m23.py** - 问题二：最终三分查找优化
   - 最终的定日镜宽度优化
   - 优化过程可视化
   - 最优布局生成

6. **m3.py** - 问题三：蒙特卡洛方法优化
   - 各定日镜安装高度优化
   - 蒙特卡洛算法实现
   - 结果对比分析

### 辅助文件
- **requirements.txt** - 依赖库列表
- **test_dependencies.py** - 依赖库测试脚本

## 运行方法

### 方法1：使用完整路径运行
```powershell
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m11.py
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m12.py
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m21.py
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m22.py
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m23.py
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe m3.py
```

### 方法2：激活虚拟环境后运行
```powershell
# 激活虚拟环境
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/Activate.ps1

# 然后可以直接使用python命令
python m11.py
python m12.py
python m21.py
python m22.py
python m23.py
python m3.py
```

## 运行建议顺序
1. 首先运行 `test_dependencies.py` 确保所有依赖正常
2. 按顺序运行问题求解文件：
   - m11.py → m12.py (问题一)
   - m21.py → m22.py → m23.py (问题二)
   - m3.py (问题三)

## 注意事项
1. **中文字体警告**: Matplotlib可能会显示中文字体缺失警告，这不影响程序运行，只是图表中的中文可能显示为方框
2. **数据文件**: 某些脚本可能需要附件数据文件，请确保相关数据文件在同一目录下
3. **输出文件**: 程序会生成多个.pkl文件和图像文件，用于保存计算结果和可视化图表
4. **计算时间**: 某些优化算法可能需要较长时间运行，特别是蒙特卡洛方法

## 故障排除
如果遇到导入错误，请运行：
```powershell
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/python.exe -c "import numpy, matplotlib, scipy; print('所有库导入成功')"
```

如果需要重新安装依赖：
```powershell
C:/Users/11312/Documents/source/py/23A/.venv/Scripts/pip.exe install -r requirements.txt
```
