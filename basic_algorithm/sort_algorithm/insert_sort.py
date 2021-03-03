"""
默认升序:
直接插入排序的核心思想是：从只包含一个数据元素的有序序列开始，不断地将待排序数据元素有序地插入这个有序序列中，直到有序序列包含了所有待排序数据元素为止。
默认有序表中包含第一个元素,那么从第二个元素开始逐个插入到有序表的正确位置，即外层循环是从第二个元素开始
"""
def insert_sort(sequence):
    for i in range(1,len(sequence)):
        j = i
        while j > 0:
            if sequence[j] < sequence[j - 1]:
                sequence[j],sequence[j-1] = sequence[j-1],sequence[j]
                j -= 1
            else:
                break
    return sequence
print(insert_sort([1,3,22,32,11,23,12,11,45,67]))
#它是稳定的，比如这里面的两个“11”,第一个11放好后，轮到第二个11排序时绝不会越过他
"""
插入排序最好的时间复杂度是正序，比较了n-1次故为O(n)，最坏是逆序比较了1+2+...+n-1次故为O(n^2)，平均次数假设
每次比较到中间，那么为O(n^2)；空间复杂度为O(1)
"""