from utils import *

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
# This function takes input data, trains a Decision Tree model using grid search,
# and returns the prediction for the input data.
def predict (input):

    # Train the model
    heart_disease = pd.read_csv("/workspaces/health_app/heart_statlog_cleveland_hungary_final.csv")

    features = heart_disease.drop('target', axis=1)
    target = heart_disease['target']

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, shuffle=True, random_state=42)

    # decision trees
    model = DecisionTreeClassifier(random_state=1)
    
    # grid search
    param_grid = {'max_depth': list(range(1,21)), 'criterion': ['entropy','gini'] }
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(features_train, target_train)
    print("Grid Search: best parameters: {}".format(grid.best_params_))
    
    # accuracy of best model with confidence interval
    best_model_DT = grid.best_estimator_
    
    pred_train_DT = best_model_DT.predict(features_train)
    pred_test_DT = best_model_DT.predict(features_test)
    
    acc_train = accuracy_score(target_train, pred_train_DT)
    acc_test = accuracy_score(target_test, pred_test_DT)
    
    print("Training Accuracy: {:3.2f}".format(acc_train))
    print("Testing Accuracy: {:3.2f}".format(acc_test))
    lb,ub = classification_confint(acc_test,features_train.shape[0])
    print("Accuracy: {:3.2f} ({:3.2f},{:3.2f})".format(acc_test,lb,ub))
    
    # build the confusion matrix
    labels = ['0','1']
    cm = confusion_matrix(target_test, pred_test_DT, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix:\n{}".format(cm_df))

    print(best_model_DT.predict(input))

    # return the prediction
    return best_model_DT.predict(input)
