#类变量（class variable）是与类关联的变量，而不是与类的实例关联，并且可供类的所有实例访问。

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

circle1 = Circle(1)
circle2 = Circle(2)

"""
上面Circle这个类里面定义的pi就是类变量
"""
#你可以这样直接用类名访问
print(Circle.pi)
#也可以从实例的__class__变量中获取类，然后在访问
print(circle1.__class__.pi)
"""
如果你用下面方式访问,也得到了相同的输出，其实程序第一步寻找的是实例变量pi，只是没找到，所以才继续找到了类变量pi
"""
print(circle1.pi)

#下面我们定义一个实例变量pi，跟类变量pi同名了，但是它们是完全不同的两个变量
circle1.pi = 3.14
print(circle1.pi)
#因为实例circle2并没有定义实例变量pi，所以你直接用circle2访问pi时，访问的其实是类变量
print(circle2.pi)
print(Circle.pi)

