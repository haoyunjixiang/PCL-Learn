## 智能指针
1. auto_ptr
2. unique_ptr
3. shared_ptr  pcl 中使用此种
4. weak_ptr
## static
1. 修饰变量 只初始化一次，存储区域由栈到堆，范围由全局到文件
2. 修饰函数 范围由全局到文件
3. 修饰类变量 属于类共有，而不是类对象私有，且需要在类外初始化，那么需要public。
4. 修饰类函数 防止修改类成员变量
## const
1. 常修饰函数引用参数，做到参数高效传递
2. 修饰局部变量，存放在堆栈区中，会分配内存（也就是说可以通过地址间接修改变量的值），全局const变量存放在只读数据段（不能通过地址修改，会发生写入错误
3. 修饰类变量，可以在构造函数初始化
4. 修饰类函数， 防止函数修改对象内容，不能与static共用。
4. const放在*左边，一律是不能修改指针内容数据，可以修改指针地址。
4. const放在*右边，一律是不能修改指针地址，可以修改指针内容数据。
## STL相关
### vector
1. 增 push_back(), c++11后emplace_back效率更高，对于大量添加用此方法。指定位置添加v.insert(v.begin()+2,3)

2. 删除earse， 两种

   a. 删除指定单个元素 v1.erase(find(v1.begin(),v2.begin(),3));

   b. 删除范围内元素 v1.erase(remove(v1.begin(), v1.end(), 3), v1.end());
   c.  v.pop_back()尾部元素删除
   d. 删除所有 v.clear()
   
3. 改 v[i] = 3;

4. 查，find方法或遍历查询

5. 初始化
	+ 拷贝初始化
	//针对的是变量
	vector<T> v2(v1);//v2中包含v1所有元素的副本
	vector<T> v2=v1;//等价于v2(v1),v2中包含有v1所有元素的副本
	+ 列表初始化（花括号）
	  //针对的是变量的元素
	  vector<T> v3{a,b,c...};//v3包含了初始值个数的元素，每个元素被赋予相应的初始值
	  vector<T> v3={a,b,c};//等价于上一个
	+ 值初始化
	  vector<T> v4(n,val);*//v5中包含着n个重复的，值为val的元素* 
	  vector<T> v5(n);*//v5包含了n个重复的，默认值的元素*
6. resize() 与reserve()
	+ resize() 在原有空间上改变
	+ reserve()重新开辟新空间，并拷贝原有内容

## 右值引用与移动语义

本质上都是为提高程序效率而设计的。如函数返回值，则需要内部先创建，返回后再销毁。如果直接赋给左边的变量是不是就更高效了呢？

1. vstr.emplace_back(std::move(str)); 在vector和string这个场景，加个`std::move`会调用到移动语义函数，避免了深拷贝。std::move就是将左值转为右值引用。这样就可以重载到移动构造函数了，移动构造函数将指针赋值一下就好了，不用深拷贝了，提高性能。
```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <string.h>
#include <string>
#include <time.h>

using namespace std;
class B
{
public:
    string str = "B";
};

class A
{
public:
    int *p = new int;
    string str = "Fuck";
    B b;
};
int main()
{
    print(get_vector());

    A a;
    A b(move(a));

    cout << a.p << " " << a.str << " " << a.b.str << " " << endl;
    cout << b.p << " " << b.str << " " << b.b.str << " " << endl;

    vector<A> v;
    clock_t s = clock();
    for (int i = 0; i < 1000000; i++)
    {
        A t;
        v.emplace_back(move(t));
    }
    clock_t e = clock();
    cout << e - s << endl;
    return 0;
}
```

## 完美转发
1. 问题：左值右值在函数调用传参时，都转化成了左值，使得函数转调用时无法判断左值和右值。（函数返回值为右值）
2. T && 为万能引用，无论左值还是右值都能匹配
	+ 若万能引用得到左值，则T = int &, 则 T &&折叠为 int &
	+ 若万能引用得到右值，则T = int， 则 T && 为 int &&
3. std::forward 与万能引用关联，原输入为左值还原为左值，若为右值则还原为右值。


```c++
template<typename T>
void print(T & t){
    std::cout << "Lvalue ref" << std::endl;
}

template<typename T>
void print(T && t){
    std::cout << "Rvalue ref" << std::endl;
}

template<typename T>
void testForward(T && v){ 
    print(v);//v此时已经是个左值了,永远调用左值版本的print
    print(std::forward<T>(v)); //本文的重点
    print(std::move(v)); //永远调用右值版本的print
std::cout << "======================" << std::endl;
}

int main(int argc, char * argv[])
{
    int x = 1;
    testForward(x); //实参为左值
    testForward(std::move(x)); //实参为右值
}

运行结果：
Lvalue ref
Lvalue ref
Rvalue ref
======================
Lvalue ref
Rvalue ref
Rvalue ref
======================
```

## C++ 赋值是浅拷贝还是深拷贝

1. =号赋值是浅拷贝，浅拷贝与深拷贝的区别在于，深拷贝会将堆上的资源数据也复制一份。而C++中指针指向的数据存在堆上，对象数据存在栈上（使用new创建会在堆上），因此使用指针时必须使用深拷贝。

   ```c++
   #include <iostream>
   
   class MyClass {
   public:
       int* data;
   
       MyClass(int value) {
           data = new int(value);
       }
   
       ~MyClass() {
           delete data;
       }
   };
   
   int main() {
       MyClass obj1(5);
       MyClass obj2 = obj1; // 赋值操作
   
       std::cout << "obj1.data: " << *obj1.data << std::endl;
       std::cout << "obj2.data: " << *obj2.data << std::endl;
   
       // 修改 obj1.data 的值
       *obj1.data = 10;
   
       std::cout << "obj1.data: " << *obj1.data << std::endl;
       std::cout << "obj2.data: " << *obj2.data << std::endl;
   
       return 0;
   }
   
   ```

   如果需要进行深拷贝，即每个对象都拥有独立的内存空间，可以自定义拷贝构造函数和赋值操作符重载函数，进行手动的深拷贝操作。
   
2. 不同赋值操作函数

   ```c++
   #include <iostream>
   #include <functional>
   using namespace std;
   
   class Test
   {
   private:
       int *data;
       int s_size;
   public:
       Test(size_t size):s_size(size){
           data=new int[s_size];
           cout<<data<<endl;
       }
       Test(Test & t){//②
   
           data=new int[t.s_size];
           cout<<"&&&"<<data<<endl;
       }
       void operator=(const Test t){
           if(t.data==nullptr){
               cout<<"error"<<endl;
               return;
           }
           if(data!=nullptr){
              delete [] data;
           }
           data=new int[t.s_size];
           cout<<"====="<<endl;
       }
       void data_addr(){
           cout<<data<<endl;
       }
       void output(int x, int y)
       {
           cout << "x: " << x << ", y: " << y << endl;
       }
       int m_number = 100;
       ~Test(){
           delete [] data;
       }
   };
   
   int main(int argc,char *argv[])
   {
       cout<<1<<endl;
       Test t(100);//调用有参构造函数
       cout<<2<<endl;
       Test b=t;//调用类引用构造函数
       b.data_addr();
       cout<<3<<endl;
       Test c(b);//调用类引用构造函数
       c.data_addr();
       cout<<4<<endl;
       Test d(200);//调用有参构造函数
       cout<<5<<endl;
   
       b.data_addr();
       cout<<6<<endl;
       d=b;//调用=号重载函数
       d.data_addr();
       cout<<7<<endl;
       b.data_addr();
       
       return 0;
   }
   ```

3. 创建对象的方式

   ```c++
   #include <iostream>  
   using namespace std;  
   class  Test {   
     private:  
     public:  
         add()
         {
            int x,y,sum;
            x=5;
            y=5;
            sum=x+y;
            cout<<sum<<endl;
        }
    };  
    void main()  
    {  
       Test test1;              //栈中分配  ，由操作系统进行内存的分配和管理
       Test test2 = Test();       //栈中分配  ，由操作系统进行内存的分配和管理
       Test *test3=new Test();  //堆中分配  ，由管理者进行内存的分配和管理，用完必须delete()，否则可能造成内存泄漏
       test1.add();
       test2.add();             //"." 是结构体成员引用
       test3->add();            //"->"是指针引用
       delete(test3);
       system("pause"); 
   }
   ```

   




## C++ 进阶书籍推荐

1. 《Effective C++》https://zhuanlan.zhihu.com/p/613356779
2. 《Effective modern C++》https://zhuanlan.zhihu.com/p/592921281
