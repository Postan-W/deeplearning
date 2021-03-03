#斐波那契数列,函数的功能是给出序列第n个数的值。递归方法


#递归方法,不是标准2^n，比这个值小，但是也属于指数型复杂度，随着n增加快速增加

k1 = 0
def fibonacci_recurence(n):
    global k1
    if n == 1 or n == 2:
        return 1
    else:
        k1 += 2
        return fibonacci_recurence(n-1) + fibonacci_recurence(n-2)

#迭代方法，复杂度线性的比如下面就是3*n
k2 = 0
def fibonacci_iteration(n):
    global k2
    if n == 1 or n == 2:
       return 1
    else:
        a,b = 1,1
        for _ in range(3,n+1):
           c = a + b
           a = b
           b = c
           k2 += 3

        return c

print("递归法结果：%d  复杂度：%d"%(fibonacci_recurence(30),k1))
print("迭代法结果：%d  复杂度：%d"%(fibonacci_iteration(30),k2))
#可以看出迭代法比递归法高效很多