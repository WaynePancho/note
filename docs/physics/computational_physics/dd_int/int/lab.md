# <center>求积分

> 使用梯形法则和辛普森法则的渐近误差公式，估算下列积分在给定精度$E$下的子分割数$n$。


$$
I_1 = \int_1^3 \mathrm{d}x \log(x),\quad \epsilon=10^{-8}
$$

$$
I_2 = \int_{-1}^{1} \mathrm{d}x e^{-x^2},\quad \epsilon=10^{-10}
$$

$$
I_3 = \int_{1/2}^{5/2} \mathrm{d}x \frac{1}{1+x^2},\quad \epsilon=10^{-12}
$$

梯形法则的子分割数为

$$
n = \sqrt{\frac{(b-a)^3 \max|f^{\prime\prime}(x)|}{12\epsilon}}
$$

辛普森（三分之一）法的子分割数为

$$
n = \sqrt[4]{\frac{(b-a)^5 \max|f^{(4)}(x)|}{90\epsilon}}
$$

为了程序简洁，使用`sympy`库来计算本题

```{.python .copy}
import sympy as sp
```

定义自变量和三个被积函数

```{.python .copy}
x = sp.symbols('x')

f1 = sp.log(x)
f2 = sp.exp(-x**2)
f3 = 1 / (1 + x**2)
```

求二阶导和四阶导{.python .copy}

```{.python .copy}
f1_2nd_deriv = f1.diff(x, 2)
f2_2nd_deriv = f2.diff(x, 2)
f3_2nd_deriv = f3.diff(x, 2)

f1_4th_deriv = f1.diff(x, 4)
f2_4th_deriv = f2.diff(x, 4)
f3_4th_deriv = f3.diff(x, 4)
```

定义积分区间和精度

```{.python .copy}
intervals = [(1, 3), (-1, 1), (1/2, 5/2)]
errors = [10**-8, 10**-10, 10**-12]
```

寻找二阶导和四阶导的最大值

```{.python .copy}
max_f1_2nd = max(abs(f1_2nd_deriv.subs(x, val)) for val in intervals[0])
max_f2_2nd = max(abs(f2_2nd_deriv.subs(x, val)) for val in intervals[1])
max_f3_2nd = max(abs(f3_2nd_deriv.subs(x, val)) for val in intervals[2])

max_f1_4th = max(abs(f1_4th_deriv.subs(x, val)) for val in intervals[0])
max_f2_4th = max(abs(f2_4th_deriv.subs(x, val)) for val in intervals[1])
max_f3_4th = max(abs(f3_4th_deriv.subs(x, val)) for val in intervals[2])
```

代入子分割数公式

```{.python .copy}
n_trapezoidal = [
    ((intervals[0][1] - intervals[0][0])**3 * max_f1_2nd / (12 * errors[0]))**(1/2),
    ((intervals[1][1] - intervals[1][0])**3 * max_f2_2nd / (12 * errors[1]))**(1/2),
    ((intervals[2][1] - intervals[2][0])**3 * max_f3_2nd / (12 * errors[2]))**(1/2)
]

n_simpson = [
    ((intervals[0][1] - intervals[0][0])**5 * max_f1_4th / (90 * errors[0]))**(1/4),
    ((intervals[1][1] - intervals[1][0])**5 * max_f2_4th / (90 * errors[1]))**(1/4),
    ((intervals[2][1] - intervals[2][0])**5 * max_f3_4th / (90 * errors[2]))**(1/4)
]
```

使用`print(n_trapezoidal, n_simpson)`打印结果，输出为

```
[8164.96580927726, 70036.1279313700, 413118.223595458]
[120.855015894271, 402.170995046513, 1349.89679420811]
```

这说明子分割数为

|       | 梯形法则 | 辛普森法则 |
| :---: | :------: | :--------: |
| $I_1$ |   8165   |    121     |
| $I_2$ |  70037   |    403     |
| $I_3$ |  413119  |    1350    |

