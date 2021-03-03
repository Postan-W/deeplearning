#生成器表达式和列表推导时很像，但是它省内存，因为它是个迭代器，返回的是一个可迭代对象，而不是整个列表

#普通列表推导
a = [i for i in range(10)]
print(a)
#output:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#字典推导
s = "abcdef"
dictionary = {s[i]:i for i in range(6)}
print(dictionary)
#生成器表达式
b = (i for i in range(10))
print(b)
print(list(b))
#output:<generator object <genexpr> at 0x000001B4152AE990>


#你可以这样用
for i in b:
    print(i)