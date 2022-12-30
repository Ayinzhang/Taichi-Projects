# Crash-balls

**碰撞球，可选择三种欧拉积分（显式，隐式，半隐式）**

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
\vec x(t_1)=\vec x(t_0)+\vec v(t_0)\Delta t
$$
*Implicit Euler's Method:*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_1)\Delta t
$$
$$
\vec x(t_1)=\vec x(t_0)+\vec v(t_1)\Delta t
$$
*Semi_Implicit Euler's Method:*
$$
\vec v(t_1)=\vec v(t_0)+M^{-1}\vec F(t_0)\Delta t
$$
$$
\vec x(t_1)=\vec x(t_0)+\vec v(t_1)\Delta t
$$



![image](https://github.com/1242857339/Taichi-simulation/blob/main/Lab1%20Crash-balls/show.png)
