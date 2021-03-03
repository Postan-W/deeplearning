"""
（1）二叉树中所有结点关键字的值都不一样；
（2）根结点如果有左子树存在，那么左子树中所有结点关键字的值都小于根结点关键字的值；
（3）根结点如果有右子树存在，那么右子树中所有结点关键字的值都大于根结点关键字的值；
（4）根结点的左子树和右子树也分别是一棵二叉搜索树。
若中序遍历这两棵二叉搜索树，那么会得到一个以结点关键字的值递增排列的有序序列，
因此，二叉搜索树也被称为二叉排序树。
不同的输入序列会导致不同形状的二叉搜索树。如果输入的序列是单调的（如1,3,5,9,12），
那么将会“退化”成一棵高度为O(n)的二叉搜索树
"""

"""
相关的有B-树、AVL二叉搜索树    
"""