"""
Huffman Tree 1952年，美国数学家哈夫曼（DavidHuffman）发明了一种二进制编码技术，被称为哈夫曼编码。在哈夫曼编码过程中使用的二叉树被称为哈夫曼树。改进后的哈夫曼树和哈夫曼编码技术已被广泛应用于数据压缩、数据传输、协同优化和大数据文本处理等领域。
"""
#除叶子结点外，其余结点都必须有两个孩子，也称为2－树
"""
树的带权路径长度（Weighted Path Length，WPL）：树中所有叶子结点的带权路径长度之和，。
其中，N是叶子结点的个数，wk是第k个叶子结点的权值，Lk是从根到该叶子结点的路径长度。
可以证明在给定节点和权重时按照哈夫曼算法构造的树是带权路径最小的
"""
