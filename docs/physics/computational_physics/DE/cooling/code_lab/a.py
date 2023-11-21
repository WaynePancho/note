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
print(r_black,r_cream)

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
