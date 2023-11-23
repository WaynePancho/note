import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

# 实验数据
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8, 51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8, 45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])
T_env = 17  # 环境温度

# 数值方法：目标函数最小化
def euler_solve(r, T_initial, time_points):
    dt = (time_points[-1] - time_points[0]) / (len(time_points) - 1)
    T = np.zeros_like(time_points)
    T[0] = T_initial
    for i in range(1, len(time_points)):
        dTdt = -r * (T[i-1] - T_env)
        T[i] = T[i-1] + dTdt * dt
    return T

def objective_function(r, T_initial, temp, time_points):
    T_model_full = euler_solve(r, T_initial, time_points)
    T_model = np.interp(time, time_points, T_model_full)
    return np.sum((temp - T_model)**2)

time_continuous = np.linspace(0, max(time), 500)  # 定义连续时间点

result_black_numeric = minimize_scalar(objective_function, args=(temp_black[0], temp_black, time_continuous), bounds=(0.001, 0.1), method='bounded')
result_cream_numeric = minimize_scalar(objective_function, args=(temp_cream[0], temp_cream, time_continuous), bounds=(0.001, 0.1), method='bounded')

r_black_numeric = result_black_numeric.x
r_cream_numeric = result_cream_numeric.x

# 线性回归方法
def estimate_r(time, temp, T_env):
    ln_temp_diff = np.log(temp - T_env)
    slope, _, _, _, _ = linregress(time, ln_temp_diff)
    return -slope

r_black_linear = estimate_r(time, temp_black, T_env)
r_cream_linear = estimate_r(time, temp_cream, T_env)

# 计算两种方法的方差
def calculate_variance(method_r, temp, T_initial):
    if method_r == 'numeric':
        r = [r_black_numeric, r_cream_numeric]
    else:
        r = [r_black_linear, r_cream_linear]

    variance_black = np.var(temp_black - euler_solve(r[0], T_initial, time))
    variance_cream = np.var(temp_cream - euler_solve(r[1], T_initial, time))

    return variance_black, variance_cream

variance_black_numeric, variance_cream_numeric = calculate_variance('numeric', temp_black, temp_black[0])
variance_black_linear, variance_cream_linear = calculate_variance('linear', temp_black, temp_black[0])

print(variance_black_numeric, variance_cream_numeric, variance_black_linear, variance_cream_linear)

