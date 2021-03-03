"""
1.如下面的函数所示，可以a:str，提示别人a参数的类型是str;这只是提示，如下所示，填入10，也不会报错，除非函数内部因为参数类型导致的错误
2.-> 类型。提示返回类型，这只是提示，如下所示提示返回列表，但是返回10并不会出错
"""
#以上叫做函数注解，可以叫做函数的元信息通过函数对象的_annotations_属性获得该信息
def getList(a:str) -> []:
    return 10
f = getList(10)
print(f)
print(getList.__annotations__)

#一个*可以将列表或元组的元素提取出作为函数的参数
a = [1,2,3]
print((lambda x,y,z:x+y+z)(*a))
#两个*可以将字典的键值提取出来作为函数的参数，键要跟函数参数能对应上
b = {"x":1,"y":2,"z":4}
print((lambda x,y,z:x+y+z)(**b))

#函数的参数
def func(*args):
    sum = 0
    for i in args:
        sum += i

    return sum
print(func(1,2,3,4,5,6))

#函数的组合
def add1(argu):
    argu += 1
    return argu

def mutiply2(argu):
    argu *= 2
    return argu

def minus5(argu):
    argu -= 5
    return argu

#定义组合器
#先介绍reduce函数
"""
reduce() 函数会对参数序列中元素进行累积。

函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
"""
from functools import reduce
def add(x, y) :            # 两数相加
    return x + y
sum1 = reduce(add, [1,2,3,4,5])   # 计算列表和：1+2+3+4+5
sum2 = reduce(lambda x, y: x+y, [1,2,3,4,5])  # 使用 lambda 匿名函数
print(sum1)
print(sum2)

#输入一个数，对这个数加一、乘2、减5一次执行
def composed(*argus):
    return reduce(lambda f,g:lambda x:f(g(x)),argus)
composed_func = composed(minus5,mutiply2,add1)

#输入5，结果应该为7，测验
print(composed_func(5))














