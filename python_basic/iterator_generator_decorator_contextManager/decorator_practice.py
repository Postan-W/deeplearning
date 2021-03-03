"""
Python装饰器的作用是使函数包装与方法包装（一个函数，接受函数并返回其增强函数）变得更容易阅读和理解
"""
"""
装饰器通常是一个命名的对象（不允许使用lambda表达式），在被（装饰函数）调用时接受单一参数，并返回另一个可调用对象。这里用的是“可调用（callable）”。而不是之前以为的“函数”。装饰器通常在方法和函数的范围内进行讨论，但它的适用范围并不局限于此。事实上，任何可调用对象（任何实现了__call__方法的对象都是可调用的）都可以用作装饰器，它们返回的对象往往也不是简单的函数，而是实现了自己的__call__方法的更复杂的类的实例
"""
"""
装饰器语法只是语法糖而已。看下面这种装饰器用法:
@some_decorator 
def function():
  pass
等价于：
def function():
  pass
decorated_function = some_decorator(function)
如果在一个函数上使用多个装饰器的话，后一种写法的可读性更差，也非常难以理解

事实上，任何函数都可以用作装饰器，因为Python并没有规定装饰器的返回类型。
因此，将接受单一参数但不返回可调用对象的函数（例如str）用作装饰器，在语法上是完全有效的。
如果用户尝试调用这样装饰过的对象，最后终究会报错。不管怎样，针对这种装饰器语法可以做一些有趣的试验
"""
#编写自定义装饰器有许多方法，但最简单的方法就是编写一个函数，返回包装原始函数调用的一个子函数
def my_decrotor(function):
    def wrapped(a):
        print("调用被包装函数")
        return function(a,a+10)

    return wrapped

@my_decrotor
def my_function(a,b):
    print("我是被包装函数")
    return a+b

print(my_function(12))

#将接收单一参数但是不返回可调用对象的任何函数用作装饰器语法上都可以，但是运行会报错
# def my_decorator2():
#     return 10
#
# @my_decorator2
# def my_function():
#     return 100
#上面报错，下面修改
# def my_decorator2(function):
#     return 10
#
# @my_decorator2
# def my_function2():
#     return 100
# #上面不报错了
# print(my_function2())
#上句报错，因为my_decorator的返回值10是不可调用的
def my_decorator2(function):
    return lambda x:x

@my_decorator2
def my_function2():
    return 100

print(my_function2(10000))
"""
简而言之，一个函数被装饰了，那么它就被作为装饰器的单一参数来使用，如上面的例子，
即便没使用也行，只要装饰器返回的是一个可调用对象就行。所以我们使用该函数名时，其实调用的是
装饰器返回的那个可调用对象

"""