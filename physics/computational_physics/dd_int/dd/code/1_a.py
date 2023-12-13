import numpy as np
import matplotlib.pyplot as plt

import matplotlib
# 设置字体为Microsoft YaHei
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Function definition
def f(x):
    return x * np.cosh(x)

# Derivative functions
def f_prime(x):
    return np.cosh(x) + x * np.sinh(x)

def f_double_prime(x):
    return 2 * np.sinh(x) + x * np.cosh(x)

# Forward difference formula for the first derivative
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

# Centered difference formula for the first derivative
def centered_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Forward difference formula for the second derivative
def forward_difference_2nd(f, x, h):
    return (f(x + 2*h) - 2*f(x + h) + f(x)) / h**2

# Centered difference formula for the second derivative
def centered_difference_2nd(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# Point of evaluation
x0 = 1.0

# Values of h
h_values = np.arange(0.05, 0.51, 0.05)

# Initialize arrays to store errors
errors_forward_first = []
errors_centered_first = []
errors_forward_second = []
errors_centered_second = []

# Calculate errors for different h values
for h in h_values:
    # First derivative
    error_forward_first = np.abs(f_prime(x0) - forward_difference(f, x0, h))
    error_centered_first = np.abs(f_prime(x0) - centered_difference(f, x0, h))
    errors_forward_first.append(error_forward_first)
    errors_centered_first.append(error_centered_first)

    # Second derivative
    error_forward_second = np.abs(f_double_prime(x0) - forward_difference_2nd(f, x0, h))
    error_centered_second = np.abs(f_double_prime(x0) - centered_difference_2nd(f, x0, h))
    errors_forward_second.append(error_forward_second)
    errors_centered_second.append(error_centered_second)

# 绘制误差图
plt.figure(figsize=(10, 6))

# 使用中文标签
plt.loglog(h_values, errors_forward_first, 'b-o', label='前向差分（一阶导数）')
plt.loglog(h_values, errors_centered_first, 'r-o', label='中心差分（一阶导数）')
plt.loglog(h_values, errors_forward_second, 'g-o', label='前向差分（二阶导数）')
plt.loglog(h_values, errors_centered_second, 'm-o', label='中心差分（二阶导数）')

# 添加中文轴标签和标题
plt.xlabel('log(h)（对数步长）')
plt.ylabel('log(Error)（对数误差）')
plt.title('数值导数在 x = 1 处的对数误差')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()


# 设定 h = 0.05
h = 0.05

# 计算一阶和二阶导数
# 使用前向差分和中心差分公式
forward_first_derivative = forward_difference(f, x0, h)
centered_first_derivative = centered_difference(f, x0, h)
forward_second_derivative = forward_difference_2nd(f, x0, h)
centered_second_derivative = centered_difference_2nd(f, x0, h)

# 显示结果
print(forward_first_derivative, centered_first_derivative, forward_second_derivative, centered_second_derivative)

# 计算精确值
exact_first_derivative = f_prime(x0)
exact_second_derivative = f_double_prime(x0)

# 显示精确值
print(exact_first_derivative, exact_second_derivative)