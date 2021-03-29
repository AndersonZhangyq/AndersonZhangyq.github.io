---
title: C++ Interview
typora-copy-images-to: C++ Interview
date: 2021-03-19 09:41:57
tags:
categories: C++
description: C++基础知识和面试题
---

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

是否内联，程序员不可控。内联函数只是对编译器的建议，是否对函数内联，决定权在于编译器。

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

