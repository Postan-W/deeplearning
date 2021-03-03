#讨论类的继承问题

class Shape:

    shape = "basic"

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def print_tag1(self):
        self.tag = "Shape"
        print(self.tag)

    def move(self,delta_x,delta_y):
        self.x += delta_x
        self.y += delta_y

class Square(Shape):

    def __init__(self,x,y,side_length):
        super().__init__(x,y)
        #上面的super()你也可以用Shape代替
        self.side = side_length

    def print_tag2(self):
        self.tag = "Square"
        print(self.tag)

square1 = Square(0,0,2)
square1.move(2,2)
print(square1.x,square1.y)

#子类和父类的同名实例变量其实用的是同一个，比如上面的self.tag
square1.print_tag1()
square1.print_tag2()

#类变量也被继承
print(Shape.shape,Square.shape,square1.shape)

#实例变量父类子类同用一个，但是类变量却不是
Square.shape = "square"
print(Shape.shape,Square.shape,square1.shape)
#并且square1这个实例也可以创建与类变量同名的实例变量
square1.shape = "instance_square"
print(Shape.shape,Square.shape,square1.shape)       