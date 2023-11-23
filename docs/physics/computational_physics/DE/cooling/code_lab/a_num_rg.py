import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 实验数据
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8, 51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8, 45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])
T_env = 17  # 环境温度

# 定义一个函数来进行线性回归并估计 r
def estimate_r(time, temp, T_env):
    ln_temp_diff = np.log(temp - T_env)
    slope, _, _, _, _ = linregress(time, ln_temp_diff)
    return -slope

# 计算黑咖啡和加奶油咖啡的冷却常数
r_black = estimate_r(time, temp_black, T_env)
r_cream = estimate_r(time, temp_cream, T_env)

# 输出结果
print(f"Estimated cooling constant for black coffee: {r_black} min^-1")
print(f"Estimated cooling constant for coffee with cream: {r_cream} min^-1")