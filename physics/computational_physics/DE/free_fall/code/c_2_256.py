import numpy as np
import matplotlib.pyplot as plt

# 设置字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
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

# 更新总时间以覆盖 n = 256 个点
n_points = 256
t_end = np.max(np.arange(1, n_points + 1) * np.pi / 4)

# 计算数值解
dt = 0.01
times, x_euler = integrate(euler_shm, dt, t_end)
_, x_cromer = integrate(euler_cromer_shm, dt, t_end)
_, x_richardson = integrate(euler_richardson_shm, dt, t_end)

# 计算精确解
t_exact = np.arange(1, n_points + 1) * np.pi / 4
x_exact = exact_solution_shm(t_exact)

# 绘制比较图
plt.figure(figsize=(12, 6))
plt.plot(times, x_euler, label='欧拉法', alpha=0.7)
plt.plot(times, x_cromer, label='欧拉-克罗默法', alpha=0.7)
plt.plot(times, x_richardson, label='欧拉-理查森法', alpha=0.7)
plt.scatter(t_exact, x_exact, color='red', label='精确解', zorder=5)
plt.xlabel('时间')
plt.ylabel('位置')
plt.title('简谐振子的位置随时间的变化（n=256）')
plt.legend()
plt.show()