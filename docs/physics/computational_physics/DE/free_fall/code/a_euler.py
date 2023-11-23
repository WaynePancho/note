import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列

# 常量
g = 9.8   # 重力加速度
y0 = 32   # 初始高度
v0 = 0    # 初始速度

# 精确解函数
def exact_solution(t):
    y_exact = y0 + v0 * t - 0.5 * g * t**2
    v_exact = v0 - g * t
    return y_exact, v_exact

# Euler 方法
def euler(y, v, dt):
    y_new = y + v * dt
    v_new = v - g * dt
    return y_new, v_new

# Euler-Cromer 方法
def euler_cromer(y, v, dt):
    v_new = v - g * dt
    y_new = y + v_new * dt
    return y_new, v_new

# Euler-Richardson 方法
def euler_richardson(y, v, dt):
    v_half = v - 0.5 * g * dt
    y_new = y + v_half * dt
    v_new = v - g * dt
    return y_new, v_new

# 使用指定的方法进行数值积分
def integrate(method, dt, t_values):
    y_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)
    y_values[0] = y0
    v_values[0] = v0

    for i in range(1, len(t_values)):
        y_values[i], v_values[i] = method(y_values[i-1], v_values[i-1], dt)

    return y_values, v_values

# 比较不同方法的精度
def compare_accuracy(dt_values):
    results = {method.__name__: [] for method in [euler, euler_cromer, euler_richardson]}
    
    for dt in dt_values:
        t_values = np.arange(0, 5, dt)
        y_exact, _ = exact_solution(t_values)  # 仅关注 y 的精确解

        for method in [euler, euler_cromer, euler_richardson]:
            y, _ = integrate(method, dt, t_values)
            error_y = np.max(np.abs(y - y_exact))  # 计算位置的最大误差
            results[method.__name__].append(error_y)
    
    return pd.DataFrame(results, index=dt_values)

# 时间步长值用于比较
dt_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# 执行精度比较
accuracy_comparison = compare_accuracy(dt_values)

# 打印精度比较结果
print(accuracy_comparison)

