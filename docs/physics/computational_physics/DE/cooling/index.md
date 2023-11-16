# <center>冷却问题

问题描述

> 若想尽快喝一杯咖啡，是立即加入奶油好，还是等一会再加？

这个问题是一个热量传递的问题，用微分方程描述咖啡的温度变化，可以写出

$$
\frac{\mathrm{d}T}{\mathrm{d}t}=f(T)
$$

即温度随时间的变化由某个函数$f(t)$给出。假设环境温度为$T_{\text{env}}$，可以预料如果咖啡的温度与环境一致，则热量就不会发生交换，从而没有热量传递，温度不变。因此写出

$$
f(T_{\text{env}}) = \left.\frac{\mathrm{d}T}{\mathrm{d}t}\right|_{T\equiv T_{\text{env}}} = 0
$$

在环境温度附近展开$f(t)$，得

$$
\frac{\mathrm{d}T(t)}{\mathrm{d}t}=f(T_{\text{env}}) + f^{\prime}(T_{\text{env}})(T-T_{\text{env}}) + \frac{1}{2!}f^{\prime\prime}(T_{\text{env}})(T-T_{\text{env}})^2 + \cdots
$$

代入$f(T_{\text{env}})=0$并略去高阶项，就可以得到牛顿冷却定律

$$
\boxed{
\frac{\mathrm{d}}{\mathrm{d} t}T(t) = r(T_{\text{env}}-T(t))
}
$$

这是一个一阶常微分方程，其常见的数值解法是欧拉算法。


