#简单选择排序，默认升序
"""
每次循环找到当前位置及以后最小的那个，然后跟这个位置交换数据
"""
def select_sort(sequence):
    for i in range(len(sequence) - 1):
        #k位置保存的是当前最小值
        k = i
        for j in range(i+1,len(sequence)):
            if sequence[j] < sequence[k]:
                k = j
        #下面交换数据
        if k == i:
            continue
        else:
            #这里也可以直接使用Python自带的交换方法
            temp = sequence[k]
            sequence[k] = sequence[i]
            sequence[i] = temp

    return sequence
a = [1,2,33,33,23,12,11,45,5,178]
print(select_sort(a))
#算法不稳定，比如第一个33和第二个33排序后顺序就颠倒了
"""
任何情况下比较次数都是n(n-1)/2,最多移动次数就是逆序即3*(n-1);空间上每次开辟一个temp空间；所以时间复杂度是
O(n^2),空间复杂度是O(1)
"""

