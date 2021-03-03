"""
stack栈是FILO先进后出结构。
queue队列是FIFO先进先出结构
用数据结构与算法高度封装Python语言反过来实现基本的数据结构有些本末倒置。
"""
class Stack:

    def __init__(self,n):
        #n表示栈的大小
        self.volume = n
        self.sequence = []
        self.top = 0
        #self.bottom = 0

    def pop_from(self):
        if self.top == 0:
            return "栈空"
        else:
            self.top -= 1
            return self.sequence.pop()

    def add_to(self,x):
        if self.top > self.volume:
            print("栈满",end=" ")
            return
        else:
            self.sequence.append(x)
            self.top += 1

stack = Stack(10)
stack.add_to(10)
stack.add_to(1)
stack.add_to(2)
for i in range(20):
    stack.add_to(i)

#够找一个循环队列
class Queue:

    def __init__(self,n):
        #n为队列大小
        self.head = 0
        self.rear = 0
        self.sequence = []
        for i in range(n):
            self.sequence.append(0)
        self.volume = n

    def add_to(self,x):
        if (self.rear + 1)%self.volume == self.head:
            print("队满")

        else:
            self.sequence[self.rear] = x
            self.rear = (self.rear + 1)%self.volume

    def pop_from(self):
        if self.head == self.rear:
            return "队空"
        else:
            a = self.sequence[self.head]
            self.head = (self.head + 1)%self.volume
            return a

queue = Queue(6)
for i in range(6):
    queue.add_to(i)
for i in range(6):
    print(queue.pop_from(),end=" ")

