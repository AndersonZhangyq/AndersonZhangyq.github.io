---
title: Interview
typora-copy-images-to: Interview
date: 2021-03-19 09:41:57
tags:
categories:
    - ["C++"]
description: 基础知识和面试题
---

# C++

## const

`Constant object can only call const member function`

```c++
class A
{
public:
    // 构造函数
    A() { };

    // const可用于对重载函数的区分
    int getValue() {
        cout << "a" << endl;
    }
    int getValue() const {
        cout << "b" << endl;
    };
};

int main() {
    const A a;
    a.getValue(); // output: b
    A b;
    b.getValue(); // output: a
}
```

## inline

是否内联，程序员不可控。内联函数只是对编译器的**建议**，是否对函数内联，决定权在于编译器。

虚函数也可以是内联函数

```c++
#include <iostream>  
using namespace std;
class Base
{
public:
    inline virtual void who()
    {
        cout << "I am Base\n";
    }
    virtual ~Base() {}
};
class Derived : public Base
{
public:
    inline void who()  // 不写inline时隐式内联
    {
        cout << "I am Derived\n";
    }
};

int main()
{
    // 此处的虚函数 who()，是通过类（Base）的具体对象（b）来调用的，编译期间就能确定了，所以它可以是内联的，但最终是否内联取决于编译器。 
    Base b;
    b.who();

    // 此处的虚函数是通过指针调用的，呈现多态性，需要在运行时期间才能确定，所以不能为内联。  
    Base *ptr = new Derived();
    ptr->who();

    return 0;
} 
```

## C++ 内存分配

通常一个由 C/C++ 编译的程序占用的内存分为以下 5 个部分:

1. 栈区（stack）: 由编译器自动分配释放，存放函数的参数值，局部变量的值等。其操作方式类似于数据结构中的栈。
2. 堆区（heap）: 一般由程序员分配释放，若程序员不释放，程序结束时可能由OS回收。注意它与数据结构中的堆是两回事，分配方式倒是类似于链表。
3. 全局区（静态区）（static）: 全局变量和静态变量的存储是放在一块的，初始化的全局变量和静态变量在一块区域，未初始化的全局变量和未初始化的静态变量在相邻的另一块区域。程序结束后由系统释放。
4. 文字常量区: 常量字符串等只读数据放在这里的。程序结束后由系统释放。
5. 程序代码区: 存放函数体的二进制代码。

[漫谈C++内存分配与管理](http://whatbeg.com/2019/04/16/cppmemory.html)

## C++中堆和栈的区别

管理方面，需要自己分配/清除
空间大小方面，堆最大可达4G（32位），而栈大小有限制，一般8M
碎片方面：堆分配和回收一段时间后可能产生碎片，栈一定不会
生长方向：栈往低地址生长，堆往高地址生长
分配方式：栈可动态分配也可静态分配，堆只能动态分配
分配效率：栈是机器系统提供的数据结构，而堆是语言层提供的数据结构，效率不一样

栈其实要比堆快，原因在于：

1. 栈是本着LIFO原则的存储机制, 对栈数据的定位相对比较快速, 而堆则是随机分配的空间, 处理的数据比较多, 无论如何, 至少要两次定位.
2. 栈是由CPU提供指令支持的, 在指令的处理速度上, 对栈数据进行处理的速度自然要优于由操作系统支持的堆数据.
3. 栈是在一级缓存中做缓存的, 而堆则是在二级缓存中, 两者在硬件性能上差异巨大.
4. 各语言对栈的优化支持要优于对堆的支持, 比如swift语言中, 三个字及以内的struct结构, 可以在栈中内联, 从而达到更快的处理速度.

## 内存池 STL
