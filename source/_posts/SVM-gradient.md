---
title: SVM 梯度
date: 2020-02-11 12:13:54
tags:
categories: cs231n
description: SVM梯度计算
---

# SVM 损失函数的梯度

训练样本$X=[x_1,x_1,\cdots,x_N]^T$，$x_i\in\mathbb{R}^{D}$，样本的类别为$Y=[y_1,y_2,\cdots,y_n]$。需要训练的参数$W=[w_1,w_2,\cdots,w_c]$，$w_i\in\mathbb{R}^{D}$。对每个样本$x_i$，SVM 的损失函数`Hinge Loss`计算方式如下：

$$
\begin{align*}
g_{ic} & = x_iw_c-x_iw_{y_i}+1 \\
L_{x_i} & =\sum_{c\ne y_i}\max(0,g_{ic}) \\
\end{align*}
$$

若使用梯度下降法来更新参数，就需要求出$\mathrm{d}W = \nabla_wL_{x_i}$，并通过迭代式$w^{t+1}=w^t+\alpha\cdot\mathrm{d}W$来更新参数。

$$
\nabla_WL_{x_i}=
\begin{bmatrix}
   \frac{\partial L_{x_i}}{\partial w_{11}} & \frac{\partial L_{x_i}}{\partial w_{12}} & \cdots & \frac{\partial L_{x_i}}{\partial w_{1C}} \\
   \vdots & \vdots &   &  \vdots  \\
   \frac{\partial L_{x_i}}{\partial w_{D1}} & \frac{\partial L_{x_i}}{\partial w_{D2}} & \cdots & \frac{\partial L_{x_i}}{\partial w_{DC}} \\
\end{bmatrix}
$$

显然，只需要算出其中每一个$\frac{\partial L_{x_i}}{\partial w_{dc}}$就可以得到梯度。先从简单地开始看，首先计算一下$\frac{\partial L_{x_i}}{\partial w_{11}}$。$w_{11}$只会出现在$x_iw_1$中，并且根据内积的运算法则，我们可以很容易得到

$$
x_iw_1=\begin{bmatrix}
   x_{i1} & x_{i2} & \cdots & x_{id}
\end{bmatrix}
\begin{bmatrix}
   w_{11} \\
   w_{21} \\
   \vdots \\
   w_{d1}
\end{bmatrix}=\sum_k{x_{ik}\cdot w_{k1}}
$$

所以有

$$
\frac{\partial g_{11}}{\partial w_{11}}=x_{i1}
$$

根据链式法则，有

$$
\begin{align*}
\frac{\partial L_{x_i}}{\partial w_{11}}&=\frac{\partial L_{x_i}}{\partial \sum}\frac{\partial \sum}{\partial \max(0,g_{11})}\frac{\partial \max(0,g_{11})}{\partial  g_{11}}\frac{\partial  g_{11}}{\partial w_{11}} \\
\\
&=\unicode{x1D7D9} \left((x_iw_1-x_iw_{y_i}+1)>0\right)x_{i1}
\end{align*}
$$

再计算$\frac{\partial L_{x_i}}{\partial w_{1c}}(c\ne y_i)$，易知

$$
\frac{\partial L_{x_i}}{\partial w_{\color{red}{1}\color{green}{c}}}=\unicode{x1D7D9} \left((x_iw_\color{green}{c}-x_iw_{y_i}+1)>0\right)x_{i\color{red}{1}}
$$

再计算$\frac{\partial L_{x_i}}{\partial w_{2c}}(c\ne y_i)$，易知

$$
\frac{\partial L_{x_i}}{\partial w_{\color{red}{2}\color{green}{c}}}=\unicode{x1D7D9} \left((x_iw_\color{green}{c}-x_iw_{y_i}+1)>0\right)x_{i\color{red}{2}}
$$

再计算$\frac{\partial L_{x_i}}{\partial w_{d1}}(1\ne y_i)$，易知

$$
\frac{\partial L_{x_i}}{\partial w_{\color{red}{d}\color{green}{1}}}=\unicode{x1D7D9} \left((x_iw_\color{green}{1}-x_iw_{y_i}+1)>0\right)x_{i\color{red}{d}}
$$

依此类推，可得

$$
\nabla_WL_{x_i}=
\begin{bmatrix}
   \unicode{x1D7D9} \left((x_iw_1-xw_{y_i}+1)>0\right)x_{i1} & \cdots & \unicode{x1D7D9} \left((x_iw_c-xw_{y_i}+1)>0\right)x_{i1} \\
   \vdots & &  \vdots  \\
   \unicode{x1D7D9} \left((x_iw_1-xw_{y_i}+1)>0\right)x_{id} & \cdots & \unicode{x1D7D9} \left((x_iw_c-xw_{y_i}+1)>0\right)x_{id} \\
\end{bmatrix}
$$

下面讨论$c=y_i$的情况。首先计算一下$\frac{\partial L_{x_i}}{\partial w_{1y_i}}$，

$$
\begin{align*}
\frac{\partial L_{x_i}}{\partial w_{1y_i}}&=\frac{\partial L_{x_i}}{\partial \sum}\frac{\partial \sum}{\partial w_{1y_i}} \\
\\
&=\sum_{c\ne y_i}\unicode{x1D7D9} \left((x_iw_c-x_iw_{y_i}+1)>0\right)\cdot(-x_{i1})
\end{align*}
$$

显然

$$
\begin{align*}
\frac{\partial L_{x_i}}{\partial w_{\color{red}{d}\color{green}{y_i}}}&=\frac{\partial L_{x_i}}{\partial \sum}\frac{\partial \sum}{\partial w_{dy_i}} \\
\\
&=\sum_{c\ne y_i}\unicode{x1D7D9} \left((x_iw_c-x_iw_\color{green}{y_i}+1)>0\right)\cdot(-x_{i\color{red}{d}})
\end{align*}
$$

对于所有样本$X$的损失函数可表示为

$$
L=\frac{1}{N}\sum_{i}^N{L_{x_i}}=\frac{1}{N}\sum_{i}^N{\sum_{c\ne y_i}\max(0,g_{ic})}
$$

所以

$$
\mathrm{d}W=\sum_i^N{\nabla_WL_{x_i}}
$$
