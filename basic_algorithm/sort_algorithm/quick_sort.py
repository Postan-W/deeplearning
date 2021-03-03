"""
快速排序算法之所以命名为“快速”，是因为到目前为止，它依然是平均情况下最快的内排序算法
快速排序的思想非常简洁：从序列中选择一个作为分割数，假定升序排列，则大于它的放右边，小于它的放到左边。
接下来左边和右边分别做同样的操作。虽说如此，操作起来快速排序还是有很多细节要注意的。
"""
def quick_sort(lis,low,high):
    #因为low和high这两个变量递归的时候还要用，所以不要改变它们
    first = low
    last = high
    # 将第一个位置作为分割值更方便更安全。无论你用哪个元素作为分割值，从算法的角度来讲对应的位置相当于空出来了。
    division = lis[first]
    #递归的时候，low和high这两个指标会相遇，那就退出，否则根据算法可知，程序就会无穷进行下去
    if low >= high:
        return

    #下面开始算法关键部分
    while first < last:
        #从后往前遍历，直到遇到比分割值小的元素，然后把这个元素和分割值掉换位置。表达式里同时取等号会出错
        while first < last and division < lis[last]:
            last -= 1
        lis[first] = lis[last]
        while first < last and division >= lis[first]:
            first += 1
        lis[last] = lis[first]

    #此时first和last是一样的，用哪个都行
    lis[last] = division
    if last - 1 > low:
        quick_sort(lis,low,last - 1)
    if last + 1 < high:
        quick_sort(lis,last + 1,high)

a = [4,1,3,2,0,-4,22,34,56,1423132,78,23,1,23,45,55]
quick_sort(a,0,len(a) - 1)
print(a)
"""
可以设想两个相同元素,一个被last扫描到了，因为小于分割值所以被被放到了左侧，而之后first扫描经过第二个相同元素，因为
小于分割值的条件故未动，但此时它们的位置相反了；所以快速排序不稳定；
最好和平均情况下的时间复杂度是O(n×log2n)；
最坏情况：在待排序序列已经排好序的情况下，其递归树成为单支树，每次划分只得到一个比上一次少一个记录的子序列。这样，必须经过n−1趟才能将所有记录定位，而且第i趟需要经过n−i次比较。O(n^2)
最好和平均情况下，栈所需耗费空间复杂度为O(log2n)，而在最坏情况下，栈所需空间为O(n)。
"""


