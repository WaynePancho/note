import numpy as np
import matplotlib.pyplot as plt

# 设置字体，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 简谐振子问题的参数
g = 9.8
y0 = 1.1  # 初始位置
v0 = 0.0  # 初始速度
omega = np.sqrt(g)  # 角频率

# 简谐振子的精确解
def exact_solution(t, y0=y0, omega=omega):
    return y0 * np.cos(omega * t)

# 欧拉法
def euler(x, v, dt):
    dxdt = v
    dvdt = -omega**2 * x
    x_new = x + dxdt * dt
    v_new = v + dvdt * dt
    return x_new, v_new

# 欧拉-克罗默法
def euler_cromer(x, v, dt):
    dvdt = -omega**2 * x
    v_new = v + dvdt * dt
    x_new = x + v_new * dt
    return x_new, v_new

# 欧拉-理查森法
def euler_richardson(x, v, dt):
    dxdt = v
    dvdt = -omega**2 * x
    x_mid = x + dxdt * dt / 2
    v_mid = v + dvdt * dt / 2
    dxdt_mid = v_mid
    dvdt_mid = -omega**2 * x_mid
    x_new = x + dxdt_mid * dt
    v_new = v + dvdt_mid * dt
    return x_new, v_new

# 数值积分
def integrate(method, y0, v0, dt, t_end):
    t = np.arange(0, t_end, dt)
    y = np.zeros(len(t))
    v = np.zeros(len(t))
    y[0], v[0] = y0, v0

    for i in range(1, len(t)):
        y[i], v[i] = method(y[i-1], v[i-1], dt)

    return t, y, v

# 时间范围和步长
t_end = 8 * np.pi
dt = 0.01

# 使用不同的方法进行数值积分
t_euler, y_euler, v_euler = integrate(euler, y0, v0, dt, t_end)
t_cromer, y_cromer, v_cromer = integrate(euler_cromer, y0, v0, dt, t_end)
t_richardson, y_richardson, v_richardson = integrate(euler_richardson, y0, v0, dt, t_end)

# 精确解
t_exact = np.linspace(0, t_end, 1000)
y_exact = exact_solution(t_exact)

plt.figure(figsize=(12, 6))

# 位置-时间图
plt.subplot(1, 2, 1)
plt.plot(t_exact, y_exact, label='精确解', linestyle='--', linewidth=2, color='black')
plt.plot(t_euler, y_euler, label='欧拉法', color='blue')
plt.plot(t_cromer, y_cromer, label='欧拉-克罗默法', linestyle='-', linewidth=2, color='red')
plt.plot(t_richardson, y_richardson, label='欧拉-理查森法', linestyle=':', color='blue')
plt.xlabel('时间')
plt.ylabel('位置')
plt.title('位置-时间图')
plt.legend()

# 速度-时间图
plt.subplot(1, 2, 2)
plt.plot(t_exact, -y_exact * omega, label='精确解', linestyle='--', color='black')
plt.plot(t_euler, v_euler, label='欧拉法', color='blue')
plt.plot(t_cromer, v_cromer, label='欧拉-克罗默法', linestyle='-', linewidth=2, color='red')
plt.plot(t_richardson, v_richardson, label='欧拉-理查森法', linestyle=':', color='blue')
plt.xlabel('时间')
plt.ylabel('速度')
plt.title('速度-时间图')
plt.legend()

plt.tight_layout()
plt.show()
