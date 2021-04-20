---
title: Noise2Noise
typora-copy-images-to: Noise2Noise
date: 2021-04-01 14:56:14
tags:
categories:
---

## 理论背景

如果有一组不可靠的观测$x=\{x_1,x_2,x_3, \cdots, x_n\}$，如何去预测真实值$z$呢？上述预测问题可以转化为一个优化问题，定义一个合适的指标$M$，那么
$$
z = \arg\min_z E_y\{M(z,x_i)\}
$$
比如线性回归的话，$M$就是平方差函数。

推广一下，把深度学习模型表示为函数$f_{\theta}$，模型的损失函数为$L$，训练集可表示为$\{(x_1, y_1), \cdots, (x_n, y_n)\}$，其中$y_i$表示gt，那么
$$
\theta = \arg\min_{\theta}E_{(x,y)}(L(f_{\theta}(x), y))
$$
那有没有可能去掉这个y？数学上可以表示为
$$
\theta = \arg\min_{\theta}E_x(E_{y|x}(L(f_{\theta}(x), y)))
$$
