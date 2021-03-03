"""
这个模块名使用的是class_overwrite类重写，只是为了起一个易懂的通俗名称
实际上我们想要做的可能是override或者overload,即对类方法的重写或重载以达到特殊目的
Python中的所有类型都是类，如果要对内置类型的行为做出修改,比如，可能要考虑由该类型派生子类
"""
#下面我们队列表派生一个类,这个类只允许列表包含初始化时候的那种元素类型
class TypedList(list):

    def __init__(self,initial_type):
        self.type = initial_type

    def __check(self,element):
        if type(element) != self.type:
            raise TypeError("the correct type is %s,but the given one is %s"%(self.type,type(element)))

    def __setitem__(self, key, value):
        self.__check(value)
        super().__setitem__(key,value)

a = TypedList(int)
a.append(1)
#下面采用赋值的方法给a添加新值，python会调用__setitem__特殊方法属性
a[0] = 1.1
