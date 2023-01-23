# Crash-balls

**回溯碰撞解除的碰撞球模型，可选择五种积分方式（显式欧拉，隐式欧拉，半隐式欧拉，速度韦尔莱，二阶龙格库塔）**

**~~但作者本人表示看不出啥区别~~**



*鼠标左键 添加碰撞球*

*w/s 增加/减少重力分量*

*d/a 增加/减少风力分量*

*↑/↓ 增加/减少潮湿度数*

*→/← 增加/减少风力分量*

*q 切换欧拉算法*

*r 清除所有碰撞球*

*Esc 退出程序*



*Explicit Euler's Method:*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_0)\Delta t
$$
$$
x(t_1)=x(t_0)+\vec v(t_0)\Delta t
$$
*Implicit Euler's Method:*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_1)\Delta t
$$
$$
x(t_1)=x(t_0)+\vec v(t_1)\Delta t
$$
*Semi_Implicit Euler's Method:*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_0)\Delta t
$$
$$
x(t_1)=x(t_0)+\vec v(t_1)\Delta t
$$

*Velocity Welley*
$$
x(t_1)=x(t_0)+\vec v(t_0)\Delta t+\frac{1}{2}M^{-1}\vec F(t_0)\Delta t^2
$$

$$
\vec v(t_1)=\vec v(t_0)+\frac{1}{2}M^{-1}(\vec F(t_0)+\vec F(t_1))\Delta t
$$

*Runge Kutta2*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_0)\Delta t
$$

$$
x(t_{0.5})=x(t_0)+\frac{1}{2}\vec v(t_0)\Delta t
$$

$$
\vec v(t_{0.5})=\vec v(t_0)+\frac{1}{2}M^{-1}\vec F(t_{0.5})\Delta t
$$

$$
x(t_1) = x(t_0)+\vec v(t_{0.5})\Delta t
$$



<div align=center>
<img src="https://github.com/1242857339/Taichi-simulation/blob/main/Lab1%20Crash-balls/show.png" width = "50%" height = "50%" />
</div> 
