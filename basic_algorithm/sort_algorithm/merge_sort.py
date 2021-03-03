"""
合并排序
分：把序列二分，对二分的序列继续二分直到子序列为1；
合：把子序列合并，合并后的序列再合并，最后得到一个有序整体
为什么最终结果有序？
在合并操作中的两个序列本身是有序的，在经过合并操作成为一个更大的有序序列。这是因为，只有一个元素时可以看作是有序的，
2 = 1+1,3=1+2,4=2+2....这样更大的序列由有序小序列合成，最终都会变成有序的
"""
def divide(sequence):
    if len(sequence) <= 1:
        return sequence

    middle = len(sequence)//2

    left = divide(sequence[:middle])
    right = divide(sequence[middle:])

    return  merge(left,right)

def merge(s1,s2):
    s3 = []
    i,j = 0,0
    #每次循环i,j都对应这s1,s2最小的元素，选出他们中更小的
    while i < len(s1) and j < len(s2):
        if s1[i] <= s2[j]:
            s3.append(s1[i])
            i += 1
        else:
            s3.append(s2[j])
            j += 1
    #循环结束就意味着有一个序列已经到头了
    if i == len(s1):
        for v in s2[j:]:
            s3.append(v)
    else:
        for v in s1[i:]:
            s3.append(v)

    return s3

def merge_sort(sequence):
    return divide(sequence)
print(merge_sort([1,22,12,11,34,543,234,23]))
"""
关于稳定性，其实这也取决于实际上你的算法怎么写，比如s1[i] <= s2[j]这个等号去掉那么就会导致算法不稳定，但一般认为
归并是稳定的算法，所以这里取等号
"""
"""
长度为n1和n2(n1≥n2)的相邻子序列进行两路合并时，比较次数至少为n2次，最多为n1+n2–1次,这个很好理解。
两路合并排序算法的最好、最坏和平均情况下的时间复杂度都是O(n×log2n)。两路合并排序最后一趟所需临时存储空间最大，
为n个单位，所以两路合并排序算法空间复杂度是O(n)。
"""