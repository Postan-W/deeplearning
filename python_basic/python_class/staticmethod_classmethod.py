#静态方法与类方法
"""
类方法与静态方法很相像，都可以在类的对象被实例化之前进行调用，也都能通过类的实例来调用。但是类方法隐式地将所属类作为
第一个参数进行传递。
利用类方法而不是静态方法，可以不必将类名硬编码写入total_area。有子类继承时，调用该类方法时，传入的类变量cls是子类，而非父类
"""
class Circle:
    pi = 3.14159
    circles = []

    def __init__(self,radius):
        self.radius = radius
        Circle.circles.append(self)

    def area(self):
        return Circle.pi*self.radius*self.radius

    @staticmethod
    def total_area():
        total = 0
        for c in Circle.circles:
            total += c.area()
        return total

    @classmethod
    def total_area2(cls):
        total = 0
        for c in cls.circles:
            total += c.area()
        return total



circle1 = Circle(1)
circle2 = Circle(2)

#下面调用下静态函数看一下我们定义的两个圆的总面积是多少
print(Circle.total_area())
#调用类方法
print(Circle.total_area2())