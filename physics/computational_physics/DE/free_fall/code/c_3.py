import numpy as np
import matplotlib.pyplot as plt

import matplotlib

# 设置字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 定义初始条件和常量
x0 = 1.1
v0 = 0
omega = 1  # 由于 k/m = 1

# 精确解函数
def exact_solution_shm(t):
    return x0 * np.cos(omega * t)

# 欧拉法
def euler_shm(x, v, dt):
    v_new = v - omega**2 * x * dt
    x_new = x + v * dt
    return x_new, v_new

# 欧拉-克罗默法
def euler_cromer_shm(x, v, dt):
    v_new = v - omega**2 * x * dt
    x_new = x + v_new * dt
    return x_new, v_new

# 欧拉-理查森法
def euler_richardson_shm(x, v, dt):
    a_half = -omega**2 * x
    v_new = v + a_half * dt
    x_new = x + 0.5 * (v + v_new) * dt
    return x_new, v_new

# 积分函数
def integrate(method, dt, t_end):
    times = np.arange(0, t_end, dt)
    x_values = np.zeros_like(times)
    v_values = np.zeros_like(times)
    
    x_values[0], v_values[0] = x0, v0

    for i in range(1, len(times)):
        x_values[i], v_values[i] = method(x_values[i-1], v_values[i-1], dt)

    return times, x_values

# 计算在 t = π 时的误差
def error_at_pi(method, dt):
    times, x_values = integrate(method, dt, np.pi)
    x_numeric_at_pi = x_values[-1]
    x_exact_at_pi = exact_solution_shm(np.pi)
    error = np.abs(x_exact_at_pi - x_numeric_at_pi)
    return error

# 不同的 Delta t 值，包括更小的步长
dt_values = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

# 记录每个 Delta t 下的误差
errors_euler = [error_at_pi(euler_shm, dt) for dt in dt_values]
errors_cromer = [error_at_pi(euler_cromer_shm, dt) for dt in dt_values]
errors_richardson = [error_at_pi(euler_richardson_shm, dt) for dt in dt_values]

# 绘制误差-步长图
plt.figure(figsize=(10, 6))
plt.plot(dt_values, errors_euler, label='欧拉法', marker='o')
plt.plot(dt_values, errors_cromer, label='欧拉-理查森法', marker='x')
plt.plot(dt_values, errors_richardson, label='欧拉-克罗默法', marker='^')
plt.axhline(y=1e-5, color='r', linestyle='--', label='误差 $10^{-5}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('时间步长 $\Delta t$')
plt.ylabel('误差')
plt.title('不同时间步长下的误差比较')
plt.legend()
plt.grid(True)
plt.show()
