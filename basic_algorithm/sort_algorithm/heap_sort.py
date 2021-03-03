"""
堆排序
堆是一棵完全二叉树，完全二叉树的特点如下：
1）它的父节点若存在，父节点的index为n//2或(n//2)-1
2）若是左子节点存在，index为2*n,序号从0开始则是2*n+1
3）若是右子节点存在，index为2*n+1,序号从0开始则是2*n+2
大根堆：每个结点的值都大于或等于左右子结点
小根堆：每个结点的值都小于或等于左右子结点

建立大根堆要逆序建立，如果正序就错了，因为从0,1,2...n-1开始建立，那么较大值只会在父亲和孩子之间选出，但是这个
较大值"不会往上爬"。而逆序即n-1,n-2,...0每一步确定了上层即父亲比下层即孩子大
"""
#先创建一个节点交换数据的函数
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

#下面定义堆排序
def heap_sort(sequence):
    n = len(sequence)

    #构建大根堆,以0为二叉树的根

    for i in range(n-1, -1, -1):
        node_exchange(sequence, n, i)

        # 一个个交换元素
    for i in range(n - 1, 0, -1):
        sequence[i], sequence[0] = sequence[0], sequence[i]
        node_exchange(sequence, i, 0)
    return sequence

print(heap_sort([1,2,22,11,8,7,6,34,56]))
"""
稳定性：可以假设两个相同元素a,b，前后关系也是a,b;他们并不是兄弟，a元素和其父元素并没有调换，b的父元素比a靠前，
并且b比其父元素大所以调换了，那么此时b就跑到a前面了。所以不稳定。
复杂度：log以2为底。初始建堆时最坏是O(n)，最好是序列本身就是大根堆形式，可以看做是0，排序时最坏是O(n*logn)即每次循环每个节点都会下沉到底部，平均也是这样，总的来说时间复杂度是O(n*logn)。
空间复杂度：每次循环一个临时空间用于进行元素交换即O(1)s
"""