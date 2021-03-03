#二叉树
class BinaryTree:
    root = None
    def __init__(self,data):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.parents = None

    @staticmethod
    def create_tree(nodes):
        #生成的形状是预定好的
        BinaryTree.root = nodes[0]
        BinaryTree.root.lchild = nodes[1]
        BinaryTree.root.rchild = nodes[2]
        nodes[1].lchild = nodes[3]
        nodes[1].rchild = nodes[4]
        nodes[2].rchild = nodes[5]
        nodes[3].lchild = nodes[6]
        nodes[4].rchild = nodes[7]
        nodes[5].lchild = nodes[8]

    @staticmethod
    def pre_order(root):
        if root == None:
            return
        print(root.data,end=" ")
        BinaryTree.pre_order(root.lchild)
        BinaryTree.pre_order(root.rchild)

    @staticmethod
    def in_order(root):
        if root == None:
            return
        BinaryTree.in_order(root.lchild)
        print(root.data,end=" ")
        BinaryTree.in_order(root.rchild)

    @staticmethod
    def last_order(root):
        if root == None:
            return
        BinaryTree.last_order(root.lchild)
        BinaryTree.last_order(root.rchild)
        print(root.data,end=" ")
nodes = []
for i in range(9):
    nodes.append(BinaryTree(i))

BinaryTree.create_tree(nodes)
BinaryTree.pre_order(BinaryTree.root)
print()
BinaryTree.in_order(BinaryTree.root)
print()
BinaryTree.last_order(BinaryTree.root)
