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

# Function to predict heart disease risk using SVM model
# This function takes input data, trains a SVM model using grid search,
# and returns the prediction for the input data.
def predict (input):

    # Train the model
    heart_disease = pd.read_csv("/content/heart_statlog_cleveland_hungary_final.csv")

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

    # return the prediction
    return best_model.predict(input)