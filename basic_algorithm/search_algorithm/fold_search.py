#折半查找,
import numpy as np
a = [1,2,3,4,5,6,11,22,33,44,55,66,12,23,34,45,56,67,78,89,90]
#时间复杂度是以2为底，序列长度为变量的
def fold_search(source,target):
    low = 0
    high = len(source) - 1
    while low <= high:
        middle = int((low + high)/2)
        if source[middle] == target:
            print("done")
            return
        elif source[middle] < target:
            low = middle + 1
        elif source[middle] > target:
            high = middle - 1

fold_search(a,55)