"""
中缀表达式是普遍使用的书写形式，但在编译程序中常用的是后缀表达式。后缀表达式是把操作符放在两个操作数之后的表达式，
又称为逆波兰表达式。其根本原因是后缀表达式中没有括号，计算时也不需要考虑操作符的优先级，
因此更易于计算机中的编译程序处理。例如:a + (b*c + d)/e对应的后缀表达式就是abc*d+e/+#,其中#是结束标识
。具体计算时使用的是栈的方法,位于左右侧的分别是左右操作数
"""
#下面利用堆栈的思想实现后缀表达式

#给出后缀表达式
s = "123*9+5/+#"
def suffix(s):
    # 构建一个栈，python中用列表代替
    a = []
    #把操作符列出来
    operator = ['+','-','*','/',"#"]
    for i in range(0,len(s)):
        if s[i] not in operator:
            a.append(int(s[i]))
        else:
            if s[i] == operator[0]:
                sum = a[-1] + a[-2]
                a.pop()
                a.pop()
                a.append(sum)
            if s[i] == operator[1]:
                subtract = a[-2] - a[-1]
                a.pop()
                a.pop()
                a.append(subtract)
            if s[i] == operator[2]:
                multiplying = a[-2] * a[-1]
                a.pop()
                a.pop()
                a.append(multiplying)
            if s[i] == operator[3]:
                divided = a[-2] / a[-1]
                a.pop()
                a.pop()
                a.append(divided)
            if s[i] == operator[4]:
                print("the result is",a[0])
suffix(s)