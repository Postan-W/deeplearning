#私有变量和私有方法只能在数据体内部使用

class Basic:

    __b = 0
    def __init__(self,value):
        self.__a = value

    def __priv_method(self):
        print(self.__a)

    def publi_method(self):
        Basic.__priv_method(self)

basic1 = Basic(1)
# print(basic1.__a)报错
# print(Basic.__a)报错
# basic1.__priv_method()报错
basic1.publi_method()
basic2 = Basic(2)
basic2.publi_method()
#看看类变量的表现
# print(Basic.__b)报错