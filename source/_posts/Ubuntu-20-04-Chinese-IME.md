---
title: Ubuntu 20.04 中文输入法
typora-copy-images-to: Ubuntu 20.04 Chinese IME
date: 2020-07-20 20:09:09
tags:
categories: Ubuntu
---

搜狗输入法在 Ubuntu 20.04 上没法用了（因为依赖了Qt4？），试了试Google拼音输入法，一言难尽，没有词库，打个字是真的累，想了想，还是再去找找其他输入法吧。

这时候百度输入法出现了，虽然没有说支持20.04，但看网上很多人说都可以用，就下下来试一试。

下载的zip包里还很贴心地配上了安装说明，可是看到说要装一大堆东西`fcitx-bin fcitx-table fcitx-config-gtk fcitx-config-gtk2 fcitx-frontend-all qt5-default qtcreator qml-module-qtquick-controls2`，这也太多了吧？？？还要装`qtcreator`？我不信要那么多。电脑上已经有fcitx了，直接安装不就好了，安装过程很顺利，但是吧，输入的前五个字母都是对的，超过5个字母，下面的候选字就变成了乱码:)

一顿操作，找到了`/opt/apps/com.baidu.fcitx-baidupinyin/files/bin/bd-qimpanel.watchdog.sh`，显然是个守护程序，打开发现守护的是同目录下的`baidu-qimpanel`。合理，来运行看看，报错:)，找不到`libQt5QuickWidgets.so.5`
行，我给你装上，`sudo apt install libqt5quickwidgets5`，装完之后可以运行了，界面还挺漂亮的，我们再试试输入法，卧槽？行了？可以，看来是缺少运行库的问题。

综上，如果安装完百度输入法乱码，其他常规fictx乱码的解决方法无效，那试试运行`/opt/apps/com.baidu.fcitx-baidupinyin/files/bin/baidu-qimpanel`，看看是不是缺少了依赖，缺什么装什么。
