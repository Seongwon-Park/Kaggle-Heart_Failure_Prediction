# Import the library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_dataset():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return df

# Scale the numeric attribute using standard scaler
def standard_scaling(df, scaling_list):
    standard_model = StandardScaler().fit(df[scaling_list])
    standard_df = standard_model.transform(df[scaling_list])
    standard_df = pd.DataFrame(standard_df)
    standard_df.columns = scaling_list
    standard_df.insert(1, 'anaemia', df['anaemia'])
    standard_df.insert(3, 'diabetes', df['diabetes'])
    standard_df.insert(5, 'high_blood_pressure', df['high_blood_pressure'])
    standard_df.insert(9, 'sex', df['sex'])
    standard_df.insert(10, 'smoking', df['smoking'])
    standard_df.insert(12, 'DEATH_EVENT', df['DEATH_EVENT'])
    return standard_df

# Scale the numeric attribute using minmax scaler
def minmax_scaling(df, scaling_list):
    minmax_model = MinMaxScaler().fit(df[scaling_list])
    minmax_df = minmax_model.transform(df[scaling_list])
    minmax_df = pd.DataFrame(minmax_df)
    minmax_df.columns = scaling_list
    minmax_df.insert(1, 'anaemia', df['anaemia'])
    minmax_df.insert(3, 'diabetes', df['diabetes'])
    minmax_df.insert(5, 'high_blood_pressure', df['high_blood_pressure'])
    minmax_df.insert(9, 'sex', df['sex'])
    minmax_df.insert(10, 'smoking', df['smoking'])
    minmax_df.insert(12, 'DEATH_EVENT', df['DEATH_EVENT'])
    return minmax_df

# Scale the numeric attribute using robust scaler
def robust_scaling(df, scaling_list):
    robust_model = RobustScaler().fit(df[scaling_list])
    robust_df = robust_model.transform(df[scaling_list])
    robust_df = pd.DataFrame(robust_df)
    robust_df.columns = scaling_list
    robust_df.insert(1, 'anaemia', df['anaemia'])
    robust_df.insert(3, 'diabetes', df['diabetes'])
    robust_df.insert(5, 'high_blood_pressure', df['high_blood_pressure'])
    robust_df.insert(9, 'sex', df['sex'])
    robust_df.insert(10, 'smoking', df['smoking'])
    robust_df.insert(12, 'DEATH_EVENT', df['DEATH_EVENT'])
    return robust_df

# Split the dataset to the target attribute and variable attribute
def split_dataset(df):
    X = df.drop(columns = ["DEATH_EVENT"], axis = 1)
    y = df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Build a gaussian nb model
def gaussian_model(X_train, X_test, y_train, y_test):
    gaussian_classifier = GaussianNB()
    parameters = {'var_smoothing' : np.logspace(0, -9, num=200)}
    gaussian_grid_search = GridSearchCV(estimator = gaussian_classifier, param_grid = parameters, scoring = 'accuracy')
    gaussian_grid_search.fit(X_train, y_train)
    best_parameter = gaussian_grid_search.best_params_
    best_score = round(gaussian_grid_search.best_score_, 4)
    print('Gaussian NB Best Parameter: {}'.format(best_parameter))
    print('Gaussian NB Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Build a logistic regression model
def logistic_model(X_train, X_test, y_train, y_test):
    logistic_classifier = LogisticRegression()
    parameters = {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [1000, 10000, 100000]}
    logistic_grid_search = GridSearchCV(estimator = logistic_classifier, param_grid = parameters, scoring = 'accuracy')
    logistic_grid_search.fit(X_train, y_train)
    best_parameter = logistic_grid_search.best_params_
    best_score = round(logistic_grid_search.best_score_, 4)
    print('Logistic Regression Best Parameter: {}'.format(best_parameter))
    print('Logistic Regression Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Build a decision tree model
def decision_model(X_train, X_test, y_train, y_test):
    decision_classifier = DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 10, 15, 30, 50, 100]}
    decision_grid_search = GridSearchCV(estimator = decision_classifier, param_grid = parameters, scoring = 'accuracy')
    decision_grid_search.fit(X_train, y_train)
    best_parameter = decision_grid_search.best_params_
    best_score = round(decision_grid_search.best_score_, 4)
    print('Decision Tree Best Parameter: {}'.format(best_parameter))
    print('Decision Tree Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Build a svm model
def svm_model(X_train, X_test, y_train, y_test):
    svm_classifier = SVC()
    parameters = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [0.01, 0.1, 1.0, 10.0]}
    svm_grid_search = GridSearchCV(estimator = svm_classifier, param_grid = parameters, scoring = 'accuracy')
    svm_grid_search.fit(X_train, y_train)
    best_parameter = svm_grid_search.best_params_
    best_score = round(svm_grid_search.best_score_, 4)
    print('SVM Best Parameter: {}'.format(best_parameter))
    print('SVM Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Visualize the confusion matrix
def visual_confusion_roc(scaler, model, params, score, X_train, X_test, y_train, y_test):
    if(model == 'GaussianNB'):
        classifier = GaussianNB(var_smoothing = params['var_smoothing'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    elif (model == 'Logistic Regression'):
        classifier = LogisticRegression(C = params['C'], max_iter = params['max_iter'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    elif (model == 'Decision Tree'):
        classifier = DecisionTreeClassifier(criterion = params['criterion'], max_depth = params['max_depth'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    elif (model == 'SVM') :
        classifier = SVC(C = params['C'], kernel = params['kernel'], gamma = params['gamma'], probability=True)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    else :
        print('ERROR:: Invalid Model Input')
    matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(matrix, annot=True, linewidth=0.7, linecolor='black', fmt='g', cmap="BuPu")
    plt.title('{0} {1} Confusion Matrix (score: {2})'.format(scaler, model, score))
    plt.xlabel('Y predict')
    plt.ylabel('Y test')
    plt.savefig('Photos/{0}_{1}_Confusion_Matrix_{2}.png'.format(scaler, model, score))
    plt.clf()
    acc_score = metrics.accuracy_score(y_test, y_predict)	
    rec_score = metrics.recall_score(y_test, y_predict)	
    pre_score = metrics.precision_score(y_test, y_predict)	
    f1s_score = metrics.f1_score(y_test, y_predict)
    print("* Accuracy: {}".format(round(acc_score, 4)))
    print("* Precision: {}".format(round(pre_score, 4)))
    print("* Recall: {}".format(round(rec_score, 4)))
    print("* F1 score: {}".format(round(f1s_score, 4)))
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label = 'ANN')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('{0} {1} ROC curve (score: {2})'.format(scaler, model, score))
    plt.savefig('Photos/{0}_{1}_ROC_curve_{2}.png'.format(scaler, model, score))
    plt.clf()

# Main Function
if __name__ == "__main__":
    scaling_list = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    non_scaling_list = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex','smoking', 'DEATH_EVENT']
    classifier_model_list = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'SVM']
    scaler_list = ['Standard Scaler', 'MinMax Scaler', 'Robust Scaler']
    total_model_list = []
    total_score_list = []
    total_param_list = []
    standard_score_list = []
    standard_param_list = []
    minmax_score_list = []
    minmax_param_list = []
    robust_score_list = []
    robust_param_list = []

    # Load the dataset
    df = load_dataset()

    # Scale the numeric attributes
    standard_df = standard_scaling(df, scaling_list) 
    minmax_df = minmax_scaling(df, scaling_list) 
    robust_df = robust_scaling(df, scaling_list) 

    # Split dataset to the test and train dataset
    sX_train, sX_test, sy_train, sy_test = split_dataset(standard_df)
    mX_train, mX_test, my_train, my_test = split_dataset(minmax_df)
    rX_train, rX_test, ry_train, ry_test = split_dataset(robust_df)

    # Used dataset - Standard Scaler
    print('\n======================================== Standard Scaler ========================================')
    stand_gaussian_best_param, stand_gaussian_best_score = gaussian_model(sX_train, sX_test, sy_train, sy_test)
    standard_param_list.append(stand_gaussian_best_param)
    standard_score_list.append(stand_gaussian_best_score)
    stand_logistic_best_param, stand_logistic_best_score = logistic_model(sX_train, sX_test, sy_train, sy_test)
    standard_param_list.append(stand_logistic_best_param)
    standard_score_list.append(stand_logistic_best_score)
    stand_decision_best_param, stand_decision_best_score = decision_model(sX_train, sX_test, sy_train, sy_test)
    standard_param_list.append(stand_decision_best_param)
    standard_score_list.append(stand_decision_best_score)
    stand_svm_best_param, stand_svm_best_score = svm_model(sX_train, sX_test, sy_train, sy_test)
    standard_param_list.append(stand_svm_best_param)
    standard_score_list.append(stand_svm_best_score)
    standard_max_index = standard_score_list.index(max(standard_score_list))
    print("Scaling Method: Standard Scaler\nBest Model: {0}\nBest Parameters: {1}\nBest Score: {2}\n"
    .format(classifier_model_list[standard_max_index], standard_param_list[standard_max_index], standard_score_list[standard_max_index]))
    total_model_list.append(classifier_model_list[standard_max_index])
    total_param_list.append(standard_param_list[standard_max_index])
    total_score_list.append(standard_score_list[standard_max_index])

    # Used dataset - MinMax Scaler
    print('\n======================================== MinMax Scaler ========================================')
    minmax_gaussian_best_param, minmax_gaussian_best_score = gaussian_model(mX_train, mX_test, my_train, my_test)
    minmax_param_list.append(minmax_gaussian_best_param)
    minmax_score_list.append(minmax_gaussian_best_score)
    minmax_logistic_best_param, minmax_logistic_best_score = logistic_model(mX_train, mX_test, my_train, my_test)
    minmax_param_list.append(minmax_logistic_best_param)
    minmax_score_list.append(minmax_logistic_best_score)
    minmax_decision_best_param, minmax_decision_best_score = decision_model(mX_train, mX_test, my_train, my_test)
    minmax_param_list.append(minmax_decision_best_param)
    minmax_score_list.append(minmax_decision_best_score)
    minmax_svm_best_param, minmax_svm_best_score = svm_model(mX_train, mX_test, my_train, my_test)
    minmax_param_list.append(minmax_svm_best_param)
    minmax_score_list.append(minmax_svm_best_score)
    minmax_max_index = minmax_score_list.index(max(minmax_score_list))
    print("Scaling Method: MinMax Scaler\nBest Model: {0}\nBest Parameters: {1}\nBest Score: {2}\n"
    .format(classifier_model_list[minmax_max_index], minmax_param_list[minmax_max_index], minmax_score_list[minmax_max_index]))
    total_model_list.append(classifier_model_list[minmax_max_index])
    total_param_list.append(minmax_param_list[minmax_max_index])
    total_score_list.append(minmax_score_list[minmax_max_index])

    # Used dataset - Robust Scaler
    print('\n======================================== Robust Scaler ========================================')
    robust_gaussian_best_param, robust_gaussian_best_score = gaussian_model(rX_train, rX_test, ry_train, ry_test)
    robust_param_list.append(robust_gaussian_best_param)
    robust_score_list.append(robust_gaussian_best_score)
    robust_logistic_best_param, robust_logistic_best_score = logistic_model(rX_train, rX_test, ry_train, ry_test)
    robust_param_list.append(robust_logistic_best_param)
    robust_score_list.append(robust_logistic_best_score)
    robust_decision_best_param, robust_decision_best_score = decision_model(rX_train, rX_test, ry_train, ry_test)
    robust_param_list.append(robust_decision_best_param)
    robust_score_list.append(robust_decision_best_score)
    robust_svm_best_param, robust_svm_best_score = svm_model(rX_train, rX_test, ry_train, ry_test)
    robust_param_list.append(robust_svm_best_param)
    robust_score_list.append(robust_svm_best_score)
    robust_max_index = robust_score_list.index(max(robust_score_list))
    print("Scaling Method: Robust Scaler\nBest Model: {0}\nBest Parameters: {1}\nBest Score: {2}"
    .format(classifier_model_list[robust_max_index], robust_param_list[robust_max_index], robust_score_list[robust_max_index]))
    total_model_list.append(classifier_model_list[robust_max_index])
    total_param_list.append(robust_param_list[robust_max_index])
    total_score_list.append(robust_score_list[robust_max_index])

    print('\n=========================================== Summary ===========================================')
    total_max_index = total_score_list.index(max(total_score_list))
    print("Scaling Method: {0}\nBest Model: {1}\nBest Parameters: {2}\nBest Score: {3}"
    .format(scaler_list[total_max_index], total_model_list[total_max_index], total_param_list[total_max_index], total_score_list[total_max_index]))

    for i in range(0, 3):
        if i == 0 :
            print('\n========== Standard Scaler Evaluation ==========')
            visual_confusion_roc(scaler_list[i], total_model_list[i], total_param_list[i], total_score_list[i], sX_train, sX_test, sy_train, sy_test)
        elif i == 1 :
            print('\n========== MinMax Scaler Evaluation ==========')
            visual_confusion_roc(scaler_list[i], total_model_list[i], total_param_list[i], total_score_list[i], mX_train, mX_test, my_train, my_test)
        elif i == 2 :
            print('\n========== Robust Scaler Evaluation ==========')
            visual_confusion_roc(scaler_list[i], total_model_list[i], total_param_list[i], total_score_list[i], rX_train, rX_test, ry_train, ry_test)
        else :
            print('Out of Range!')