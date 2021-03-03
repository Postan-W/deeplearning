""""
生成器函数是内部有yield关键字的函数，因为python语言里面函数就是一个对象，
那么这种带yield关键字的函数就可以叫做生成器对象generator,
也可以叫它迭代器iterator。
"""
"""
你可以用next()函数或for循环从生成器中获取新的元素，就像迭代器一样
基于yield语句，生成器可以暂停函数并返回一个中间结果
斐波那契数列是无穷的，但用来生成它的生成器每次提供一个值，并不需要无限大的内存
实现的原理是该函数会保存执行上下文.
"""
def fibonacci():
    a,b = 0,1
    while True:
        yield b
        a,b = b,a+b
#你可以用next()函数或for循环从生成器中获取新的元素，就像迭代器一样
fib = fibonacci()
for i in range(10):
    print(next(fib))
print([next(fib) for i in range(10)])
#再python里函数也是对象，所以上面的fib对象一直在更新上下文
"""
1
1
2
3
5
8
13
21
34
55
[89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

但是如果你不小心print(next(fibnacci()))这样写,显然输出的一直是1
"""
#为了体现其可迭代性，还可以这样用
for i in fibonacci():
    if i > 200:
        break
    print(i)

"""
Python生成器的另一个重要特性，就是能够利用next函数与调用的代码进行交互。
yield变成了一个表达式，而值可以通过名为send的新方法来传递
"""
def hava_a_chat():
    print("please tell me your problem:")

    while True:
        answer = (yield)#此处接收send的值，类似于阻塞
        if answer is not None:
            if answer.endswith("?"):
                print("I don't konw")
            elif 'good' in answer:
                print('congratulations')
            elif 'bad' in answer:
                print("hope you get well soon")

chat = hava_a_chat()
next(chat)
chat.send("feel good")
chat.send("feel bad")
"""
send的作用和next类似，但会将函数定义内部传入的值变成yield的返回值。因此，这个函数可以根据客户端代码来改变自身行为。为完成这一行为，还添加了另外两个函数：throw和close。它们将向生成器抛出错误。• throw：允许客户端代码发送要抛出的任何类型的异常。• close：作用相同，但会引发特定的异常——GeneratorExit。在这种情况下，生成器函数必须再次引发GeneratorExit或StopIteration
"""