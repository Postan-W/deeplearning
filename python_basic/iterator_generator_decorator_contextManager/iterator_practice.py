"""
迭代器只不过是一个实现了迭代器协议的容器对象。
它基于以下两个方法。
• __next__：返回容器的下一个元素。
• __iter__：返回迭代器本身。
"""
"""
当遍历完序列时，会引发一个StopIteration异常。这样迭代器就可以与循环兼容，
因为可以捕获这个异常并停止循环。要创建自定义的迭代器，可以编写一个具有__next__方法的类，
只要这个类提供返回迭代器实例的__iter__特殊方法
"""
"""
__iter__返回一个可迭代的对象，如果一个类实现了这个魔法函数，那么这个类就是可迭代对象，并且实现了__next__这个魔法函数的话，可以通过for循环遍历；如果单独实现了__next__这一个魔法函数，只能通过next()调用
"""
class Count:
    def __init__(self,step):
        self.step = step

    def __next__(self):
        if self.step <= 0:
            raise StopIteration
        self.step -= 1
        return self.step

    def __iter__(self):
        return self
c = Count(10)
print(next(c),"from next()")
print(next(c),"from next()")
for i in c:
    print(i)

#例2,迭代器可以利用内置的iter函数和一个序列来创建
iteration2 = iter("abcd")
print(next(iteration2),"from next()")
for c in iteration2:
    print(c)
