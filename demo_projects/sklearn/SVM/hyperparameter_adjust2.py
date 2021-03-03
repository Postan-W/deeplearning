import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,model_selection
from sklearn.metrics import classification_report

import utilities 

# Load input data
input_file = '../files/svm_data_multivar.txt'
X, y = utilities.load_data(input_file)

###############################################
# Train test split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)

# Set the parameters by cross-validation
parameter_grid = [  {'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                 ]

metrics = ['precision', 'recall_weighted']

for metric in metrics:
    print("\n#### Searching optimal hyperparameters for", metric)

    classifier = model_selection.GridSearchCV(svm.SVC(C=1),
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nScores across the parameter grid:")
    means = classifier.cv_results_['mean_test_score']
    params =classifier.cv_results_['params']

    print(params, '-->', means)

    print("\nHighest scoring parameter set:", classifier.best_params_)

    y_true, y_pred = y_test, classifier.predict(X_test)
    print("\nFull performance report:\n")
    print(classification_report(y_true, y_pred))

