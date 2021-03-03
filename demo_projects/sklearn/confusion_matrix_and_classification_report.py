import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utility.machinelearning_matplotlib import plot_confusion_matrix

y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat)

# Print classification report
from sklearn.metrics import classification_report
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names))
print(confusion_mat)