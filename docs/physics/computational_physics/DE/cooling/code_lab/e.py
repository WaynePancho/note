import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib

# 设置字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 实验数据及环境温度
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8, 51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
T_env = 17

# 计算温度变化率
dTdt_black = np.diff(temp_black) / np.diff(time)
T_midpoints_black = (temp_black[:-1] + temp_black[1:]) / 2

# 定义一个正值拟合函数
def r_fit_positive(T, a, b):
    return a * np.exp(-b * (T - T_env))

# 对黑咖啡的数据进行拟合
params_black_positive, _ = curve_fit(r_fit_positive, T_midpoints_black, dTdt_black / (T_env - T_midpoints_black), p0=[0.1, 0.01])

# 绘制 r(T) 的图像
T_range = np.linspace(min(temp_black), max(temp_black), 100)
r_values = r_fit_positive(T_range, *params_black_positive)

plt.figure(figsize=(10, 6))
plt.plot(T_range, r_values, label="拟合的 r(T)")
plt.scatter(T_midpoints_black, dTdt_black / (T_env - T_midpoints_black), color='red', label="实验数据点")
plt.title("黑咖啡温度依赖的冷却常数 r(T)")
plt.xlabel("温度 (°C)")
plt.ylabel("冷却常数 r")
plt.legend()
plt.grid(True)
plt.show()

# 使用龙格-库塔法求解修正后的 T(t)
def runge_kutta_method_corrected(T_initial, time_points, r_func, params, T_env):
    T = np.zeros_like(time_points)
    T[0] = T_initial
    dt = time_points[1] - time_points[0]
    for i in range(1, len(time_points)):
        k1 = dt * -r_func(T[i-1], *params) * (T[i-1] - T_env)
        k2 = dt * -r_func(T[i-1] + 0.5 * k1, *params) * (T[i-1] + 0.5 * k1 - T_env)
        k3 = dt * -r_func(T[i-1] + 0.5 * k2, *params) * (T[i-1] + 0.5 * k2 - T_env)
        k4 = dt * -r_func(T[i-1] + k3, *params) * (T[i-1] + k3 - T_env)
        T[i] = T[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return T

# 使用修正的龙格-库塔法求解 T(t)
time_continuous = np.linspace(0, max(time), 500)
T_pred_black_runge_kutta_positive = runge_kutta_method_corrected(temp_black[0], time_continuous, r_fit_positive, params_black_positive, T_env)

# 绘制修正后的 T(t) 图像
plt.figure(figsize=(10, 6))
plt.plot(time_continuous, T_pred_black_runge_kutta_positive, label="修正后的数值解 T(t)")
plt.scatter(time, temp_black, color='red', label="实验数据点 (黑咖啡)")
plt.title("修正后的黑咖啡温度随时间的变化 (龙格-库塔法)")
plt.xlabel("时间 (分钟)")
plt.ylabel("温度 (°C)")
plt.legend()
plt.grid(True)
plt.show()
