---
title: ssh
typora-copy-images-to: ssh
date: 2021-11-19 08:50:22
tags:
categories:
---

# 端口映射
## 远程端口映射到本地
config文件里可以配置
```
Host {{name}}
    HostName {{remote-ip}}
    User {{username}}
    LocalForward {{local-ip}}:{{local-port}} {{remote-ip}}:{{remote-port}}
```