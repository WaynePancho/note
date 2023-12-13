import numpy as np
import matplotlib.pyplot as plt

import matplotlib

# 设置字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 定义常量和函数
g = 9.8   # 重力加速度
y0 = 32   # 初始高度
v0 = 0    # 初始速度

def exact_solution(t):
    y_exact = y0 + v0 * t - 0.5 * g * t**2
    v_exact = v0 - g * t
    return y_exact, v_exact

def euler(y, v, dt):
    y_new = y + v * dt
    v_new = v - g * dt
    return y_new, v_new

def euler_cromer(y, v, dt):
    v_new = v - g * dt
    y_new = y + v_new * dt
    return y_new, v_new

def euler_richardson(y, v, dt):
    v_half = v - 0.5 * g * dt
    y_new = y + v_half * dt
    v_new = v - g * dt
    return y_new, v_new

def integrate(method, dt, t_values):
    y_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)
    y_values[0] = y0
    v_values[0] = v0
    for i in range(1, len(t_values)):
        y_values[i], v_values[i] = method(y_values[i-1], v_values[i-1], dt)
    return y_values, v_values

# 设置时间步长
dt = 0.1
t_values = np.arange(0, 5, dt)

# 获取精确解
y_exact, v_exact = exact_solution(t_values)

# 使用不同的方法进行积分
y_euler, v_euler = integrate(euler, dt, t_values)
y_cromer, v_cromer = integrate(euler_cromer, dt, t_values)
y_richardson, v_richardson = integrate(euler_richardson, dt, t_values)

# 绘制 y(t) 图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_values, y_exact, label="精确解", linestyle='dashed', linewidth=2.5, color='black')
plt.plot(t_values, y_euler, label="欧拉法")
plt.plot(t_values, y_cromer, label="欧拉-克罗默")
plt.plot(t_values, y_richardson, label="欧拉-理查森")
plt.title("$y$ 随时间变化")
plt.xlabel("时间 $t$")
plt.ylabel("位置 $y$")
plt.legend()

# 绘制 v(t) 图
plt.subplot(1, 2, 2)
plt.plot(t_values, v_exact, label="精确解", linestyle='dashed', linewidth=2.5, color='black')
plt.plot(t_values, v_euler, label="欧拉法")
plt.plot(t_values, v_cromer, label="欧拉-克罗默")
plt.plot(t_values, v_richardson, label="欧拉-理查森")
plt.title("$v$ 随时间变化")
plt.xlabel("时间 $t$")
plt.ylabel("速度 $v$")
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

