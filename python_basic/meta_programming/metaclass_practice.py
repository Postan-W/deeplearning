"""
元类是定义其他类（型）的一种类（型）。为了理解其工作方式，最重要的是要知道，
定义了对象实例的类也是对象。因此，如果它也是对象的话，那么一定有与其相关联的类。
所有类定义的基类都是内置的type类
即：
certain instance ->is instance of ->certain class->is instance of->type
"""
"""
在Python中，可以将某个类的元类替换为我们自定义的类型。
通常来说，新的元类仍然是type类的子类（参见图3-4），因为如果不是的话，
这个类将在继承方面与其他的类非常不兼容。
用class语句创建的每个类都隐式地使用type作为其元类。
可以通过向class语句提供metaclass关键字参数来改变这一默认行为:class Myclass(metaclass=..):
即：
certain instance ->is instance of ->certain class->is instance of->certain class->is instance of->type
"""
"""
调用内置的type()类可作为class语句的动态等效。给定名称、基类和包含属性的映射，它会创建一个新的类对象
"""
def method(self):
    return 1000

kclass = type('Myclass',(object,),{'method':method})

instance = kclass()
print(instance.method())
"""
等价于
class Myclass:
   def method(self):
        return 1000
"""