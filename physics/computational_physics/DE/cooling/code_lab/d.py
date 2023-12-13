# 设置初始参数
T_initial_black = 82.3  # 黑咖啡的初始温度 (°C)
T_env = 17  # 环境温度 (°C)
dt = 0.1  # 时间步长 (分钟)
r_black = 0.02589108681175482 # 第一种数值方法解出的冷却常数 (/分钟)

# 计算温差变为原始温差的 1/2, 1/4, 1/8 时所需时间的函数
def time_for_temp_reduction(T_initial, r, reduction_factors, T_env, dt):
    times = []
    for factor in reduction_factors:
        T_target = T_env + (T_initial - T_env) / factor
        T_current = T_initial
        t = 0
        while T_current > T_target:
            T_current += -r * (T_current - T_env) * dt
            t += dt
        times.append(t)
    return times

# 计算黑咖啡冷却到不同温差所需的时间
time_reduction_factors = [2, 4, 8]
times_black = time_for_temp_reduction(T_initial_black, r_black, time_reduction_factors, T_env, dt)


print(times_black)