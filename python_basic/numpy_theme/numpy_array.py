"""
创建一个 ndarray 只需调用 NumPy 的 array 函数即可

numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
object	数组或嵌套的数列
dtype	数组元素的数据类型，可选
copy	对象是否需要复制，可选
order	创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）
subok	默认返回一个与基类类型一致的数组
ndmin	指定生成数组的最小维度
------------------------------------------------------------------
ndarray 对象由计算机内存的连续一维部分组成，并结合索引模式，将每个元素映射到内存块中的一个位置。内存块以行顺序(C样式)或列顺序(FORTRAN或MatLab风格，即前述的F样式)来保存元素。


"""
import numpy as np
a = np.array([1,2,3])
b = np.array([[1,  2],  [3,  4]])
c = np.array([1,  2,  3,4,5], ndmin =  2)
d = np.array([1,  2,  3], dtype = complex)
print(a)
print(b)
print(c)
print(d)
"""
[1 2 3]
[[1 2]
 [3 4]]
[[1 2 3 4 5]]
[1.+0.j 2.+0.j 3.+0.j]
"""