# set up
import pandas as pd
import numpy as np
import sklearn
np.set_printoptions(formatter={'float_kind':"{:3.2f}".format})
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# compute 95% confidence intervals for classification and regression
def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    import math
    interval = 1.96*math.sqrt(acc*(1-acc)/n)
    lb = max(0, acc - interval)
    ub = min(1.0, acc + interval)
    return (lb,ub)


# Function to predict heart disease risk using SVM model
# This function takes input data, trains a SVM model using grid search,
# and returns the prediction for the input data.
def predict (input):

    # Train the model
    heart_disease = pd.read_csv("/workspaces/health_app/heart_statlog_cleveland_hungary_final.csv")

    features = heart_disease.drop('target', axis=1)
    target = heart_disease['target']

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, shuffle=True, random_state=42)

    # SVM model
    model = SVC(max_iter=10000)

    # grid search
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(features_train, target_train)

    # accuracy of best model with confidence interval
    best_model = grid.best_estimator_

    pred_train_SVM = best_model.predict(features_train)
    pred_test_SVM = best_model.predict(features_test)

    acc_train = accuracy_score(target_train, pred_train_SVM)
    acc_test = accuracy_score(target_test, pred_test_SVM)

    print("Training Accuracy: {:3.2f}".format(acc_train))
    print("Testing Accuracy: {:3.2f}".format(acc_test))
    lb,ub = classification_confint(acc_test,features_train.shape[0])
    print("Accuracy: {:3.2f} ({:3.2f},{:3.2f})".format(acc_test,lb,ub))


    print(best_model.predict(input))

    # return the prediction
    return best_model.predict(input)