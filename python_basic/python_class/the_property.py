"""
在类的使用过程中出于一些考虑，你不想直接把变量暴露出去，所以你可以为该变量设置get和set方法，但是python有更简单的方法，就是用点访问一个属性，感觉就像直接使用一个变量一样简单，而且又保证了背后有一个处理过程，不想直接使用变量取或赋值那么轻率
"""
class Student(object):

    def __init__(self):
        self._score = 0
        self._age = 0
    #用property这个装饰器，下面的函数就变成了了一个属性
    @property
    def score(self):
        return self._score

    #为了实现set功能，还要用score.setter这样的形式修饰
    @score.setter
    def score(self, value):
        if value > 100:
            self._score = 100
            return
        else:
            self._score = value

    @property
    def age(self):
        return self._age

st = Student()
st.score = 200
# print(st.score)报错

#因为属性age没有setter，所以你不能给这个属性赋值
print(st.age)
st.age = 20