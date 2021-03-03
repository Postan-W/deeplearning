"""
将测试标签打乱然后匹配得到一个在此测试标签下的随机正确率,这可以作为一个对模型效率的参考。毕竟谁也不好意思让模型的正确率比随机的还低，就像你认真考试出来的结果没有别人蒙的分数高，这很尴尬。
"""
import numpy as np
import copy
def random_accuracy(test_data):
    copy_test_data = copy.copy(test_data)
    np.random.shuffle(copy_test_data)
    hits = np.array(test_data) == np.array(copy_test_data)
    accuracy = float(np.sum(hits)) / len(test_data)
    return accuracy

#测试
if __name__ == '__main__':
    print(random_accuracy([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))