---
title: 最大熵模型
date: 2018-12-04 09:46:18
tags:
categories: Machine Learning
---

# 熵

## 熵的定义

在信息论和概率统计中，熵（Entropy）是表示随机变量不确定性的度量。

$X$是一个取有限个值的离散随机变量，其概率分布为
$$
P(X=x_i)=p_i,\ i=1,2,3\cdots n
$$
则随机变量$X$的熵定义为
$$
H(X)=-\sum_{i=1}^{n}{p_i\log{p_i}}\\
p_i=0 \Rightarrow p_i\log{p_i}=0
$$
可以证明：0 \le H(X) \le \log{n}

熵的最小值是0是显然的，下面用拉格朗日乘数法来证明熵的最大值是$\log{n}$
$$
\begin{align*}
\max \ \ &H(X)\\
s.t. \ \ &\sum_{i=1}^{n}p_i=1\\
L(p_i,\omega)&=-\sum_{i=1}^{n}{p_i\log{p_i}}+\omega(\sum_{i=1}^{n}{p_i}-1)\\
let \ \ 0&=\frac{\partial L}{\partial p_i}\\
&=-\log{p_i}-p_i\frac{1}{p_i\ln{2}}+\omega\\
&=-\log{p_i}-\frac{1}{\ln{2}}+\omega\\
\therefore \log{p_i}&=-\frac{1}{\ln{2}}+\omega\\
\therefore \log{p_1}&=\log{p_2}=\cdots =\log{p_n}\\
\therefore p_1&=p_2=\cdots =p_n\\
\therefore p_i&=\frac{1}{n}
\end{align*}
$$
带入p_i=\frac{1}{n}后得到熵的最大值是\log{n}

## 条件熵

条件熵$H(Y|X)$表示在已知随机变量$X$的条件下随机变量Y的不确定性。随机变量$X$给定点条件下随机变量Y的条件熵$H(Y|X)$可表示为：
$$
H(Y|X)=-\sum_{x,y}^{}{P(x,y)\log{P(Y|X)}}
$$

# 最大熵模型

## 最大熵原理

最大熵原理是概率模型学习的一个准则。最大熵原理认为，学习概率模型时，在所有可能的概率模型（分布）中，熵最大的模型是最好的模型。通常用约束条件来确定概率模型的集合，所以，最大熵原理也可表述为在满足约束条件的模型集合中选取熵最大的模型。

在上一节中，已经证明了当随机变量$X$服从均匀分布时，熵最大。也就是说，当我们没有更多信息的情况下，那些不确定的部分都是等可能的。

## 最大熵模型的定义

将最大熵原理应用到分类得到最大熵模型

> 假设分类模型是一个条件概率分布$P(Y|X)$，$X\in \mathcal{X} \subseteq R^n$表示输入，$Y \in \mathcal{Y}$表示输出，$\mathcal{X}$和$\mathcal{Y}$分别是输入和输出的集合，这个模型表示的是对于给定点输入$X$，以条件概率$P(Y|X)$输出$Y$

给定一个训练数据集
$$
T=\{(x_1, y_1),(x_2, y_2),\cdots,(x_n, y_n)\}
$$
学习的目标是用最大熵原理选择最好的分类模型

### 经验分布

首先考虑模型应该满足的条件。给定数据集，可以确定联合分布$P(X,Y)$的经验分布$\tilde{P}(X,Y)$和边缘分布$P(X)$的经验分布$\tilde{P}(X)$
$$
\begin{align*}
\tilde{P}(X=x,Y=y)&=\frac{v(X=x,Y=y)}{N}\\
\tilde{P}(X=x)&=\frac{v(X=x)}{N}
\end{align*}
$$
其中$v(X=x,Y=y)$表示训练数据中样本$(x,y)$出现的频数，$v(X=x)$表示训练数据中输入$x$出现的频数，$N$表示训练样本容量。

由于真实的边缘分布不可知，我们使用贝叶斯公式$P(x,y)=P(y|x)P(x)$将其转化为条件概率和输入$x$的边缘分布的乘积，并使用输入$x$的经验分布来近似，由此得到模型优化的目标函数
$$
\begin{align*}
max \ \ H(Y|X)&=-\sum_{x,y}^{}{P(x,y)\log{P(Y|X)}}\\
&\approx-\sum_{x,y}^{}{\tilde{P}(x)P(y|x)\log{P(Y|X)}}
\end{align*}
$$


### 特征函数

用特征函数$f(x,y)$描述输入$x$与输出$y$之间的某一个事实，定义为：
$$
\begin{equation*}
f(x,y)=\left\{
\begin{aligned}
1, \ & if \ (x, y) \ meets \ the \ demand\\
0, \ & otherwise \\
\end{aligned}
\right.
\end{equation*}
$$
它是一个二值函数，当$x$和$y$满足这个事实时取值为1，否则取值为0

我们希望模型尽可能多的从训练数据中得到信息，于是我们假设特征函数关于经验分布$\tilde{P}(X,Y)$的期望值$E_{\tilde{p}}(f)$与其关于模型的期望值$E_p(f)$相等。

其中$E_{\tilde{p}}(f)$可表示为
$$
E_{\tilde{p}}(f)=\sum_{x,y}\tilde{P}(x,y)f(x,y)
$$
$E_p(f)$可近似为
$$
\begin{align*}
E_p(f)&=\sum_{x,y}P(x,y)f(x,y)\\
&\approx\sum_{x,y}P(y|x)P(x)f(x,y)\\
&\approx\sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)
\end{align*}
$$
由此，我们得到模型的一个约束
$$
\begin{align*}
E_p(f)&=E_{\tilde{p}}(f)\\
\sum_{x,y}\tilde{P}(x)P(y|x)f(x,y)&=\sum_{x,y}\tilde{P}(x,y)f(x,y)
\end{align*}
$$

# 最大熵模型的学习

在开始模型学习之前，我们还需要增加一个显然的约束条件：$\sum_{x,y}P(y|x)=1$。

最大熵模型的学习可以形式化为带约束条件的最优化问题：

> 给定训练数据集$T=\{(x_1, y_1),(x_2, y_2),\cdots,(x_n, y_n)\}$以及特征函数$f_i(x,y), \ i=1,2,3,\cdots m$，最大熵模型的学习等价于如下最优化问题
> $$
> \begin{align*}
> \max \ \  &-\sum_{x,y}^{}{\tilde{P}(x)P(y|x)\log{P(Y|X)}}\\
> s.t. \ \ &\sum_{y}P(y|x)=1\\
> &\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x,y)=\sum_{x,y}\tilde{P}(x,y)f_i(x,y) \ \ i=1,2,3,\cdots m
> \end{align*}
> $$
>

我们可以通过解决对偶问题的方式来进行学习

我们将上述最大化问题转化为等价的最小化问题，即在所需最大化的函数前加上负号，接着引进拉格朗日乘子$\omega_0,\omega_1,\omega_2,\cdots,\omega_m$，定义拉格朗日函数$L(P,\omega)$
$$
\begin{align*}
L(P,\omega)&=\sum_{x,y}^{}{\tilde{P}(x)P(y|x)\log{P(Y|X)}}+\omega_0\left(1-\sum_{x,y}P(y|x)\right)\\
&\ \ \ \ +\sum_{i=1}^{n}\omega_i\left(\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x,y)\right) \tag{1}
\end{align*}
$$
最优化问题的原始问题是
$$
\min_{P\in C} \max_{\omega} L(P,\omega)
$$
对偶问题是
$$
\max_{\omega} \min_{P\in C} L(P,\omega)
$$

首先求解极小化问题$\min_{P\in C} L(P,\omega)$
$$
\begin{align*}
0=\frac{\partial L(P,\omega)}{\partial P(y|x)}&=\tilde{P}(x)\left(\log{P(y|x)} + P(y|x)\frac{1}{P(y|x)}\right)-\omega_0-\sum_{i=1}^{n}\omega_i\tilde{P}(x)f_i(x,y)\\
\log{P(y|x)}&=\frac{\left(\omega_0+\sum_{i=1}^{n}\omega_i\tilde{P}(x)f_i(x,y)\right)}{\tilde{P}(x)}-1\\
P(y|x)&=\exp{\left(\sum_{i=1}^{n}\omega_i f_i(x,y)\right)}\exp\left(\frac{\omega_0}{\tilde{P}(x)}-1\right)
\end{align*}
$$

由于约束$\sum_{y}P(y|x)=1$，将上式抽象为
$$
P(y|x)=\exp{\left(\sum_{i=1}^{n}\omega_i\tilde{P}(x)f_i(x,y)\right)}\frac{1}{Z_{\omega}(x)} \tag{2}
$$
后带入约束得
$$
\begin{align*}
1&=\sum_{y}\exp{\left(\sum_{i=1}^{n}\omega_i f_i(x,y)\right)}\frac{1}{Z_{\omega}(x)}\\
Z_{\omega}(x)&=\sum_{y}\exp{\left(\sum_{i=1}^{n}\omega_i f_i(x,y)\right)} \tag{3}
\end{align*}
$$
最后将公式(2)、公式(3)带入到公式(1)中，得到所需最大化的函数
$$
\begin{align*}
L(P,\omega)=&\sum_{x,y}^{}{\tilde{P}(x)P(y|x)\log{P(y|x)}}+\omega_0\left(1-\sum_{x,y}P(y|x)\right)\\
&+\sum_{i=1}^{n}\omega_i\left(\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x,y)\right)
\end{align*}
$$

# Reference

1. [最大熵模型](https://blog.csdn.net/v_JULY_v/article/details/40508465)