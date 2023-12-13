import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import matplotlib

# 设置字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 实验数据及环境温度
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8, 51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8, 45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])
T_env = 17

# 欧拉方法求解微分方程
def euler_solve(r, T_initial, time_points):
    dt = (time_points[-1] - time_points[0]) / (len(time_points) - 1)
    T = np.zeros_like(time_points)
    T[0] = T_initial
    for i in range(1, len(time_points)):
        dTdt = -r * (T[i-1] - T_env)
        T[i] = T[i-1] + dTdt * dt
    return T

# 最小化的目标函数
def objective_function(r, T_initial, temp, time_points):
    T_model_full = euler_solve(r, T_initial, time_points)
    T_model = np.interp(time, time_points, T_model_full)  # 插值以匹配实验数据时间点
    return np.sum((temp - T_model)**2)

# 创建具有更高密度的时间点数组
time_continuous = np.linspace(0, max(time), 500) # 即确定时间步长

# 寻找冷却常数 r 的最佳估计
result_black = minimize_scalar(objective_function, args=(temp_black[0], temp_black, time_continuous), bounds=(0.001, 0.1), method='bounded')
r_black = result_black.x

result_cream = minimize_scalar(objective_function, args=(temp_cream[0], temp_cream, time_continuous), bounds=(0.001, 0.1), method='bounded')
r_cream = result_cream.x

print(r_black,r_cream)

# 使用欧拉方法计算预测温度
T_pred_black = euler_solve(r_black, temp_black[0], time_continuous)
T_pred_cream = euler_solve(r_cream, temp_cream[0], time_continuous)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.scatter(time, temp_black, color='blue', label='实验数据 (黑咖啡)')
plt.plot(time_continuous, T_pred_black, color='blue', linestyle='dashed', label=f'模拟曲线 (黑咖啡, r = {r_black:.4f})')
plt.scatter(time, temp_cream, color='red', label='实验数据 (加奶油咖啡)')
plt.plot(time_continuous, T_pred_cream, color='red', linestyle='dashed', label=f'模拟曲线 (加奶油咖啡, r = {r_cream:.4f})')
plt.title('咖啡冷却过程中的温度变化 (数值方法)')
plt.xlabel('时间 (分钟)')
plt.ylabel('温度 (°C)')
plt.legend()
plt.grid(True)
plt.show()