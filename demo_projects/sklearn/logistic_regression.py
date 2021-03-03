import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt
from utility.machinelearning_matplotlib import plot_classifier

if __name__=='__main__':
    # input data
    X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # initialize the logistic regression classifier
    #参数C表示对分类错误（misclassification）的惩罚值（penalty）。在一定范围内越大越好，可以实验c=100的分类边界比c=1明显好
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # train the classifier
    classifier.fit(X, y)

    # draw datapoints and boundaries
    plot_classifier(classifier, X, y)
