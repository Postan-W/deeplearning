"""
__method()__：特殊方法,例如__add__()、__init__()等
__attribute__:特殊属性，例如__dict__,获得对象所绑定的所有属性和方法的字典
特殊方法（special method ）对Python而言具备特殊的含义。
虽然被定义为方法，但其实并不是打算直接当作方法使用的。通常特殊方法不会被直接调用，而是由Python
自动调用，以便对属于该类的对象的某种请求做出响应。
"""
"""
__getitem__

单独实现这个魔法函数，可以让这个类成为一个可迭代的对象，
并且可以通过使用下标获取类中元素值下标的元素
"""
class IteratorClass:
    #init也是一个特殊方法
    def __init__(self,filepath):
        self.fileobject = open(filepath,'r',encoding='utf-8')

    #返回类的可读字符串表示
    def __str__(self):
        return "该类有特殊方法__getitem__，所以类对象可以被用作迭代器使用"

    def __getitem__(self, index):
        line = self.fileobject.readline()
        if line == "":
            self.fileobject.close()
            raise IndexError

        else:
            return line

"""
上面的类可以作为读取文件的工具类。一下子读取整个文件，可能内存会吃不消，所以一行一行的读一般是我们采用的
方法，直接用这个类对象进行迭代比较方便。
"""
file_iterator = IteratorClass("..\\records\\error.txt")
print(file_iterator)
#输出特殊属性
print(file_iterator.__dict__)

for line in file_iterator:
    print(line)

#__getitem__(self,index)是的该类的对象变成一个可迭代对象，其中index参数如上是默认传入的、自增的，但是面的函数实现中并没有使用index这个参数，这也是可以的

#下面看另一个例子
class Library(object):
    def __init__(self):
        self.books = [1, 2, 3]
        self.index = -1

    def __getitem__(self, i):
        return self.books[i]

l = Library()
print(l[1])
for i in l:
    print(i)
#上面再函数里没有异常处理，这也是可以的，因为列表越界后系统抛出异常，被for捕获
#与__getitem__类似的是__iter__，参见iterator专题

