"""
冒泡排序 默认升序
每次循环都是挨个交替元素位置，每次交换都是把大的放后面。所以第n次循环就会把第n大的放在倒数第n个位置上。
自然地，需要n-1次外层循环完成排序，具体来说第n-1次排序就是第一和第二个元素的排序
"""
def bubble(sequence):
    # 如果某次内层循环没有交换元素，那么就不用了排了，已经是有序的了
    flag = 1
    for i in range(1,len(sequence)):
        #如果flag为0那么就是说某次内层循环没有交换元素则排序完成
        if flag == 0:
            break
        flag = 0
        for j in range(len(sequence)-1):
            if sequence[j] > sequence[j+1]:
                sequence[j],sequence[j+1] = sequence[j+1],sequence[j]
                flag = 1
    return sequence

print(bubble([1,22,12,32,31,45,678,2345,12,33,45,56]))

"""
两个相等的元素是不会交换位置的，无论它们相邻与否，所以这是稳定算法
如果没有设置flag则无论什么情况都会进行全部的双层循环，那么时间复杂度可知为O(n^2)，但是有flag，那么本来就正序的
是最好的情况，时间复杂度是O(n)；空间复杂度就是O(1)，在其他语言中体现的更明显，因为使用了temp中间变量
"""