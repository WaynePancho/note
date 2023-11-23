# 公共变量和常数
class Common:
    g = 9.8  # 重力加速度 (m/s^2)
    y = 10   # 初始高度 (m)
    v = 0    # 初始速度 (m/s)
    a = -g   # 加速度 (m/s^2)
    t = 0    # 初始时间 (秒)

def initial():
    """ 设置初始条件和参数 """
    Common.t = 0
    Common.y = 10
    Common.v = 0
    Common.a = -Common.g
    Common.dt = float(input("时间步长 dt = "))

def print_table(nshow):
    """ 打印模拟数据 """
    if Common.t == 0:
        nshow[0] = int(input("输出间隔的时间步数 = "))
        print(f"{'时间':>8}{'y':>16}{'速度':>16}{'加速度':>14}")
        print()
    print(f"{Common.t:13.4f}{Common.y:13.4f}{Common.v:13.4f}{Common.a:13.4f}")

def euler():
    """ 执行欧拉方法的下一步 """
    Common.a = -Common.g
    Common.y += Common.v * Common.dt
    Common.v += Common.a * Common.dt
    Common.t += Common.dt

def free_fall():
    """ 模拟自由落体的主程序 """
    nshow = [0]  # 使用列表模拟按引用传递
    initial()
    print_table(nshow)
    counter = 0
    while Common.y > 0:
        euler()
        counter += 1
        if counter % nshow[0] == 0:
            print_table(nshow)
    print_table(nshow)  # 打印在地面时的值

# 运行程序
free_fall()
