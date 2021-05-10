---
title: Self-Supervised
typora-copy-images-to: Self-supervised
date: 2020-07-21 17:27:08
tags:
categories: Self-supervised
description: 自监督论文笔记
---

## CVPR 2020 SelFlow: Self-Supervised Learning of Optical Flow

### Unsupervised Optical Flow Estimation

基于亮度一致性`brightness constancy`和空间平滑性`spatial smoothness`，由此产生了`photometric loss`，但是这个损失函数并不能处理有遮挡的情况，所以后来的学者提出先估计出一个`occlusion map`，然后计算损失的时候不去考虑这些被遮挡的点。

### Self-supervised Method

思考一下，自监督怎么才能监督起来呢？已经提出了很多的`pretext task`，比如说`inpaint`（从图片中去掉一块，然后去填充）、给图片上色、或者是拼图任务。其实自监督的关键就是要从`pretext task`中学到一些特殊的信息，比如拼图任务，就是希望能学习出图片的空间位置关系。

### Method

使用两个结构一样的CNN网络，`NOC-Model`重点学习没有被遮挡的像素点的光流，`OCC-Model`学习所有像素点（包括被遮挡和不被遮挡的像素点）的光流，测试的时候只需要`OCC-Model`就可以了。

给定连续三帧$I_{t-1}$、$I_t$和$I_{t+1}$，我们用$w_{i\rightarrow j}$表示$I_{i}$到$I_j$的光流，那么我们可以估计$w_{t\rightarrow t-1}$和$w_{t\rightarrow t+1}$，然后用估计出来的两个光流把$I_t$转化为$I_{t-1}$和$I_t$，这时候就可以做自监督了。另外，为什么要估计两个光流呢，只顾及一个光流其实也够啊？作者把$w_{t\rightarrow t+1}$作为forward，把$w_{t\rightarrow t-1}$作为backward，相当于在网络的一次前向传播就完成了forward和backward两次光流估计，是个有新意的地方。

## ICML-2020 Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere

常用的 Contrastive Learning 的损失函数是 InfoNCE，即
$$
\mathcal L_{contrastive}(f;\tau,M)=E\left[-\log\frac{e^{f(x)^Tf(y)/\tau}}{e^{f(x)^Tf(y)/\tau}+\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}\right]
$$
原文中的变换实在没看懂，主要是当$P[f(x)=f(y)]=1$时如何得到那个简化形式。仔细研究后，我们可以通过另一种方式得到文中的结论，记录如下：
$$
\begin{aligned}
\mathcal L_{contrastive}(f;\tau,M)&=E\left[-\log\frac{e^{f(x)^Tf(y)/\tau}}{e^{f(x)^Tf(y)/\tau}+\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}\right] \\
&=E\left[\log\frac{e^{f(x)^Tf(y)/\tau}+\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}{e^{f(x)^Tf(y)/\tau}}\right]\\
&=E\left[1+\log\frac{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}{e^{f(x)^Tf(y)/\tau}}\right]
\end{aligned}
$$
我们的优化目标是$\min{\mathcal L_{contrastive}(f;\tau,M)}$，可得
$$
\begin{align}
\min{\mathcal L_{contrastive}(f;\tau,M)} &= \min{E\left[1+\log\frac{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}{e^{f(x)^Tf(y)/\tau}}\right]}\\
&=\min{E\left[\log\frac{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}{e^{f(x)^Tf(y)/\tau}}\right]}\\
&=\min{E\left[\log{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}-\log{e^{f(x)^Tf(y)/\tau}}\right]}\\
&=\min{E\left[\log{e^{-f(x)^Tf(y)/\tau}}+\log{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}\right]}\\
&=\min{E\left[-f(x)^Tf(y)/\tau+\log{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}\right]}\\
\end{align}
$$
所以损失函数的优化可以理解为两个部分，第一部分是最小化$-f(x)^Tf(y)/\tau$，这里点积是一个度量相似度的函数，这一部分的优化目标是让`positive pair`之间的特征尽可能的对齐（或者说相似度尽可能的大）；第二部分是最小化$\log{\sum_i{e^{f(x_i^-)^Tf(y)/\tau}}}$，也就是让`negative pair`之间的特征均匀地分散开，按照文中的逻辑，任意两个不同类别之间的特征相似度很小也就意味着很分散，分散到极致就是均匀分布了（只是一种感觉？）。