---
title: SVM
typora-copy-images-to: SVM
date: 2021-09-09 01:38:54
tags: 
categories:
    - ["Machine Learning"]
---
# 引入
SVM是一种常用的分类模型，其目标是寻找一个能将所有样本点准确划分的分离超平面（即分离超平面的两侧分别对应两个类别），优化目标是最大化几何间隔。为了能够应对线性不可分的数据集，引入了核技巧，其核心是将样本点映射到一个高维空间，认为低维空间不可分的数据集总能在高维空间中找到一个分离超平面将他们区分开来。

# 原理
假定我们有如下数据集，
$$
D=\lbrace (x_{11}, x_{12}, \cdots, x_{1d}, y_1), \cdots, (x_{n1}, x_{n2}, \cdots, x_{nd}, y_n) \rbrace
$$
数据集中每个样本$i$都有$d$个特征，$y_i$表示样本$i$所属的类别。简单起见，我们以二分类问题为例，此时有$y_i\in \lbrace0, 1\rbrace$。

## 函数间隔和几何间隔
我们用$\vec w\cdot \vec x+b$来表示一个分离超平面，并作如下规定：若$y_i=+1$，则$\vec w\cdot \vec x_i+b > 0$；若$y_i=-1$，则$\vec w\cdot \vec x_i+b < 0$。

显然，对于任意的$\vec x_i$，都有$y_i(\vec w\cdot \vec x_i+b) > 0$。这里，$y_i(\vec w\cdot \vec x_i+b)$就是函数间隔。函数间隔描述了两件事情：

1. 函数间隔前的符号反映了样本是否被正确分类。如果大于0，那么这个样本就被正确分类了；
2. 函数间隔的绝对值描述了这个样本属于当前类的确信度，绝对值越大确信度越高。

因此，我们可以用函数间隔来评价分离超平面的优劣，更进一步，我们用数据集中的最小函数间隔来评价分离超平面的优劣。我们希望找到最小函数间隔尽可能大的分离超平面。

但是函数间隔有一个问题，当我们成倍扩大$\vec w$和$b$时，函数间隔也会随之变大，这显然不是我们想要的，所以引入几何间隔的概念：
$\frac{y_i(\vec w\cdot \vec x_i+b)}{||\vec w||_2}$。可见，相比于函数间隔，几何间隔用$\vec w$的2范数做了归一化。（在几何意义上，$\frac{|\vec w\cdot \vec x_i+b|}{||\vec w||_2}$表示样本点到分离超平面的距离，这大概就是几何间隔的由来）。

给定数据集D，我们定义分离超平面关于数据集D的几何间隔为所有样本点距离分离超平面的最小几何间隔，即
$$
\gamma = \min_{i=1,2,\cdots,N}{\frac{y_i(\vec w\cdot \vec x_i + b)}{||\vec w||_2}}
$$

## 决策函数
如果我们找到了想要的超平面，即
$$
(\vec w^*, b^*) = \underset{\vec w, b}{\operatorname{argmin}}\gamma
$$
那么决策函数可以写成$sign(\vec w^*\cdot \vec x_i+b^*)$

## 支持向量和间隔
令$\beta > 0$，如果存在样本点$\vec x_j$使得下式中的等号成立
$$
\left \lbrace 
\begin{align*}
\vec w\cdot \vec x_a+b &\ge \beta,\ y_a = 1 \\
\vec w\cdot \vec x_b+b &\le -\beta,\ y_b = -1
\end{align*}
\right .
$$
那么这些样本点对应的特征向量就被称之为支持向量。间隔定义为两个异类支持向量到分离超平面的距离，即
$$
\gamma = \frac{2\beta}{||\vec w||_2}
$$

## 间隔最大化
间隔最大化意味着，找到的分离超平面能以最大的确信度将样本集分为两类。我们的优化目标可以表示为
$$
\begin{align*}
\max_{\vec w,b}\ &\frac{2\beta}{||\vec w||_2} \\
s.t.\ &y_i(\vec w\cdot \vec x_i+b) \ge \beta\ \ i = 1,2,\cdots,N
\end{align*}
$$
显然，$\beta$的取值和优化目标无关，上式可以重写为
$$
\begin{align*}
\min_{\vec w,b}\ &{\frac{1}{2}||\vec w||^2} \\
s.t.\ &y_i(\vec w\cdot \vec x_i+b) \ge 1\ \ i = 1,2,\cdots,N
\end{align*}
$$

## 对偶问题
应用拉格朗日乘子法，将约束写入到目标函数中，得到
$$
L(\vec w,b,\vec \alpha) = \frac{1}{2}||\vec w||^2 + \sum_i^N{\alpha_i(1 - y_i(\vec w\cdot \vec x_i+b))} \tag{1}
$$
原问题的解等价于
$$
\max_{\alpha}{\min_{\vec w,b}\ {L(\vec w,b,\vec \alpha)}}
$$
首先求解极小化问题，得：
$$
\begin{align*}
\triangledown L_\vec w = \vec w - \sum_i^N{\alpha_i y_i \vec x_i} \\
\triangledown L_b = -\sum_i^N{\alpha_i y_i}
\end{align*}
$$
极值一定在一阶导数导数为零处取到，因此令上式为零后，得
$$
\begin{align*}
0 = \triangledown L_b &= -\sum_i^N{\alpha_i y_i} \\
0 = \triangledown L_\vec w &= \vec w - \sum_i^N{\alpha_i y_i \vec x_i} \\
\vec w &= \sum_i^N{\alpha_i y_i \vec x_i}
\end{align*}
$$
将$\vec w = \sum_i^N{\alpha_i y_i \vec x_i}$和$0= \sum_i^N{\alpha_i y_i}$带入`(1)`得
$$
\begin{align*}
L(\vec w,b,\vec \alpha) &= \frac{1}{2}||\sum_i^N{\alpha_i y_i  \vec x_i}||^2 + \sum_i^N{\alpha_i(1 - y_i(\sum_j^N{\alpha_j y_i \vec x_j}\cdot \vec x_i+b))} \\ 
&= \frac{1}{2}||\sum_i^N{\alpha_i y_i \vec x_i}||^2 + \sum_i^N{\alpha_i} -\sum_i^N{\alpha_i y_i(\sum_j^N{\alpha_j y_j \vec x_j}\cdot \vec x_i+b))} \\ 
&= \frac{1}{2}||\sum_i^N{\alpha_i y_i \vec x_i}||^2 + \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i y_i \alpha_j y_j \vec x_j\cdot \vec x_i}}+ b\sum_i^N{\alpha_i y_i} \\ 
&= \frac{1}{2}||\sum_i^N{\alpha_i y_i \vec x_i}||^2 + \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i \alpha_j y_i y_j \vec x_i\cdot \vec x_j}} \tag{2}
\end{align*}
$$
`(2)`最后第一项和第三项，应该是不能直接合并的。当$i \ne j$时，该项存在于第三项中但是第一项里没有。但从求极值的角度来说，`(2)`等价于
$$
\begin{align*}
&\max_{\alpha}\ L(\vec w,b,\vec \alpha) \\ 
\Leftrightarrow &\max_{\alpha}\ \frac{1}{2}||\sum_i^N{\alpha_i y_i \vec x_i}||^2 + \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i \alpha_j y_i y_j \vec x_i\cdot \vec x_j}} \\ 
\Leftrightarrow &\max_{\alpha}\ \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i \alpha_j y_i y_j \vec x_i\cdot \vec x_j}} \tag{3} \\
&\ \ s.t.\ \sum_i^N{\alpha_i y_i} = 0
\end{align*}
$$

## 参数计算
由KKT条件可知，当
$$
\alpha_i(1 - y_i(\vec w \cdot \vec x_i+b)) = 0
$$
时，对偶问题和原问题取到极值。由于$\alpha_i > 0$，所以当且仅当$1 - y_i(\vec w \cdot \vec x_i+b) = 0$时上式等号成立，即$\vec x_i$为支持向量时等号成立。因此，线性SVM的解为
$$
\vec w = \sum_{i\in SV}^N{\alpha_i y_i \vec x_i} \tag{4}
$$
将`(4)`带入$1 - y_i(\vec w \cdot \vec x_i+b) = 0$，考虑到$y_i \in \lbrace -1, 1 \rbrace $，所以$1 / y_i = y_i$，得到$$
b = y_i - \vec w \cdot \vec x_i = y_i - \sum_{i\in SV}^N{\alpha_i y_i \vec x_i} \cdot \vec x_i
$$
实践中用所有支持向量算出来的b的均值作为最终解。

## 核技巧
假设样本是线性不可分的，那么线性SVM就无法找到分离超平面，此时就需要使用核技巧，简答来说就是将样本的特征空间向高维空间做映射。假设我们使用函数$\Phi$对样本的特征空间做映射$R^d \rightarrow R^{\hat{d}}$，那么`(3)`可表示为
$$
\begin{align*}
&\max_{\alpha}\ \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i \alpha_j y_i y_j \Phi(\vec x_i)\cdot \Phi(\vec x_j) }} \\
&\ \ s.t.\ \sum_i^N{\alpha_i y_i} = 0
\end{align*} \tag{5}
$$
从`(5)`中可知，即使我们不知道映射函数$\Phi$的具体形式，只需要算出$\Phi(\vec x_i)\cdot \Phi(\vec x_j)$即可。假设我们能找到函数$\kappa$满足
$$
\kappa(\vec x_i\cdot \vec x_j) = \Phi(\vec x_i)\cdot \Phi(\vec x_j)
$$
那么我们就能用函数$\kappa$实现向高维空间映射的同时还维持计算复杂度依然是$O(d)$。那么`(5)`就可写为
$$
\begin{align*}
&\max_{\alpha}\ \sum_i^N{\alpha_i} -\sum_i^N{\sum_j^N{\alpha_i \alpha_j y_i y_j \kappa(\vec x_i\cdot \vec x_j) }} \\
&\ \ s.t.\ \sum_i^N{\alpha_i y_i} = 0
\end{align*} \tag{5}
$$
常见的核函数有

||||
|----|---------|-|
|线性核|$\vec x_i^T\cdot \vec x_j$|d远大于n|
|多项式核|$(\beta \vec x_i^T\cdot \vec x_j + \theta)^n$|d较小，n中等|
|RBF核|$\exp {(-\frac{||\vec x_i - \vec x_j||^2}{2\sigma^2} )}$|d较小，n很大|