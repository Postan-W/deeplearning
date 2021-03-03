"""
heap中文叫做堆，是一颗完全二叉树,树中每个结点的值都小于或者等于其左右孩子结点的值，这样的堆称为小根堆（最小堆）；树中每个结点的值都大于或者等于其左右孩子结点的值，则称为大根堆（最大堆）。完全二叉树的根称为堆顶。堆顶结点的值是所有结点中最小或最大的。
"""
#建立大根堆
def node_exchange(sequence,n,i):
    #sequence是要排序的序列，n是下标的终止范围，i是当前节点位置
    largest = i
    l = 2*i + 1
    r = 2*i + 2
    #下面是找出节点i和它的左右孩子中的最大值，把最大值放在父亲的位置上
    if l < n and sequence[i] < sequence[l]:
        largest = l

    if r < n and sequence[largest] < sequence[r]:
        largest = r

    if largest != i:
        sequence[i],sequence[largest] = sequence[largest],sequence[i]
        node_exchange(sequence,n,largest)

sequence = [22,1,3,6,78,89,32,41]
n = len(sequence)
#注意，一定是逆序建立，不然算法是错的
for i in range(n-1, -1, -1):
    node_exchange(sequence, n, i)

print(sequence)
