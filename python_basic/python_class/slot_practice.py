"""
有一个有趣的特性几乎从未被开发人员使用过，就是槽（slots）。它允许你使用__slots__属性来为指定的类设置一个静态属性列表，并在类的每个实例中跳过__dict__字典的创建过程。它可以为属性很少的类节约内存空间，因为每个实例都没有创建__dict__
"""
class Frozen():
    __slots__ = ['ice','cream','degree']
    def __init__(self,degree):
        self.degree = degree


class Frozen2():
    def __init__(self, degree):
        self.degree = degree
f1 = Frozen(-10)
f2 = Frozen(-11)

f3 = Frozen2(-5)
f4 = Frozen2(-6)

print('__dict__' in dir(Frozen))
print('__dict__' in dir(Frozen2))