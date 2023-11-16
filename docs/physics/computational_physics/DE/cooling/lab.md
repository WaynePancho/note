# <center>Coffee cooling program

## (a) 确定冷却常数

> 根据牛顿冷却定律，找出描述以下表格中实验结果的黑咖啡和加奶油咖啡的 $r$ 近似值（从以下表格中判断 r 是否为常数）。你确定 $r$ 最佳值的隐含标准是什么？给出确定 $r$ 最佳值的两种方法，并论证哪种更好以及为什么更好。因为时间是以分钟为单位测量的，所以冷却常数 $r$ 的单位是 $\min^{-1}$。




欧拉法+最小二乘法



## (b) 可视化

> 使用在 (a) 部分中找到的 $r$ 值，绘制温度随时间变化的图表。将表中的数据绘制在同一图表上，并与你的结果进行比较。你可以使用任何图形包来绘制图表。

在本题中


![](../../../../assets/images/physics/computational_physics/DE/2.png)








## (c) 讨论时间步长

> 时间步长 $\Delta t$ 有无物理意义？确保你选择的 $\Delta t$ 足够小，以至于不会影响你的结果。你应该估算结果的误差。

123:smile:

![](../../../../assets/images/physics/computational_physics/DE/3.png)



## (d) 实例计算

> 黑咖啡与周围环境的初始温差大约为 $76{}^{\circ}\text{C}$。咖啡冷却到温差为 $76/2 = 38{}^{\circ}\text{C}$ 需要多长时间？温差变为 $76/4$ 和 $76/8$ 需要多长时间？试图在不首先使用计算机的情况下用简单的术语理解你的结果。

123



## (e) 关于牛顿冷却定律适用程度的讨论

> 参考表格，讨论牛顿冷却定律是否适用于一杯咖啡的冷却，以及可以做哪些修改？

123



## Table

玻璃杯中咖啡的温度。温度记录的估计精度为 $0.1{}^{\circ}\text{C}$。空气温度为 $17{}^{\circ}\text{C}$。第二列对应黑咖啡，第三列对应加重奶油的咖啡。

| time (min) | $T{}^{\circ}\text{C}$ (black) | $T{}^{\circ}\text{C}$ (cream) | time (min) | $T{}^{\circ}\text{C}$ (black) | $T{}^{\circ}\text{C}$ (cream) |
| :--------: | :---------------------------: | :---------------------------: | :--------: | :---------------------------: | :---------------------------: |
|     0      |             82.3              |             68.8              |     24     |             51.2              |             45.9              |
|     2      |             78.5              |             64.8              |     26     |             49.9              |             44.8              |
|     4      |             74.3              |             62.1              |     28     |             48.6              |             43.7              |
|     6      |             70.7              |             59.9              |     30     |             47.2              |             42.6              |
|     8      |             67.6              |             57.7              |     32     |             46.1              |             41.7              |
|     10     |             65.0              |             55.9              |     34     |             45.0              |             40.8              |
|     12     |             62.5              |             53.9              |     36     |             43.9              |             39.9              |
|     14     |             60.1              |             52.3              |     38     |             43.0              |             39.3              |
|     16     |             58.1              |             50.8              |     40     |             41.9              |             38.6              |
|     18     |             56.1              |             49.5              |     42     |             41.0              |             37.7              |
|     20     |             54.3              |             48.1              |     44     |             40.1              |             37.0              |
|     22     |             52.8              |             46.8              |     46     |             39.5              |             36.4              |















## <center>数值方法有效性的讨论



直接解出解析解


![](../../../../assets/images/physics/computational_physics/DE/1.png)


```{.python .copy title="直接解出理论解再进行拟合"}
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import matplotlib

# 设置字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 实验数据
# 时间 (分钟)
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
# 黑咖啡温度
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8, 51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
# 加奶油咖啡温度
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8, 45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

# 环境温度
T_env = 17

# 定义牛顿冷却定律函数
def newton_cooling(t, T_initial, r):
    return T_env + (T_initial - T_env) * np.exp(-r * t)

# 最小化的目标函数
def objective_function(r, T_initial, temp):
    return np.sum((temp - newton_cooling(time, T_initial, r))**2)

# 黑咖啡冷却常数
result_black = opt.minimize_scalar(objective_function, args=(temp_black[0], temp_black))
r_black = result_black.x

# 加奶油咖啡冷却常数
result_cream = opt.minimize_scalar(objective_function, args=(temp_cream[0], temp_cream))
r_cream = result_cream.x

# 显示求出的咖啡冷却常数
r_black, r_cream

# 使用计算出的冷却常数 r 来绘制温度随时间变化的图表

# 生成理论模型数据
time_model = np.linspace(0, 46, 100)
temp_model_black = newton_cooling(time_model, temp_black[0], r_black)
temp_model_cream = newton_cooling(time_model, temp_cream[0], r_cream)

# 绘制图表
plt.figure(figsize=(10, 6))

# 实验数据
plt.scatter(time, temp_black, color='blue', label='实验数据 (黑咖啡)')
plt.scatter(time, temp_cream, color='red', label='实验数据 (加奶油咖啡)')

# 理论模型
plt.plot(time_model, temp_model_black, color='blue', linestyle='dashed', label=f'理论模型 (黑咖啡, r = {r_black:.4f})')
plt.plot(time_model, temp_model_cream, color='red', linestyle='dashed', label=f'理论模型 (加奶油咖啡, r = {r_cream:.4f})')

plt.xlabel('时间 (分钟)')
plt.ylabel('温度 (°C)')
plt.title('咖啡冷却曲线')
plt.legend()
plt.grid(True)
plt.show()

```