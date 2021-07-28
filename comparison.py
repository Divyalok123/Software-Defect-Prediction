import pandas as pd
import numpy as np
import preprocessing as prep
import models
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, roc_auc_score, precision_score
import sys

prev_stdout = sys.stdout
f = open('_result.txt', 'w')
sys.stdout = f

def print_scores(clf_name, y_val, y_val_pred, y_test, y_test_pred):
    print(clf_name)
    print()
    print("-- Validation Set --")
    print("Accuracy: ", accuracy_score(y_val, y_val_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_val, y_val_pred))
    print("Precision: ", precision_score(y_val, y_val_pred))
    print("AUC: ", roc_auc_score(y_val, y_val_pred))
    print("Recall: ", recall_score(y_val, y_val_pred))
    print()
    print("-- Test Score --")
    print("Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_test_pred))
    print("Precision: ", precision_score(y_test, y_test_pred))
    print("AUC: ", roc_auc_score(y_test, y_test_pred))
    print("Recall: ", recall_score(y_test, y_test_pred))

    print()
    print()

datafiles = ['ar1.csv', 'ar3.csv', 'ar4.csv', 'ar5.csv', 'ar6.csv', 'cm1.csv', 'kc1.csv', 'kc2.csv', 'mc1.csv', 'mc2.csv', 'pc1.csv', 'pc2.csv', 'pc3.csv', 'pc4.csv']
filename = './Data/' + datafiles[-1]

X, y, X_train, X_test, X_validation, y_train, y_test, y_validation = prep.preprocess_data(filename, 10)

#Convolutional Neural Network
clf = models.CNN(X, X_train, X_validation, y_train, y_validation)
X_validation_matrix = X_validation.values
X_validation1 = X_validation_matrix.reshape(X_validation_matrix.shape[0], 1, len(X_validation.columns), 1)
y_val_pred = clf.predict(X_validation1) > 0.5
X_test_matrix = X_test.values
X_test1 = X_test_matrix.reshape(X_test_matrix.shape[0], 1, len(X_test.columns), 1)
y_test_pred = clf.predict(X_test1) > 0.5
print_scores("CNN", y_validation, y_val_pred, y_test, y_test_pred)

#Random Forest classifier
clf = models.random_forest(X_train, y_train)
y_val_pred = clf.predict(X_validation)
y_test_pred = clf.predict(X_test)
print_scores("RANDOM FOREST", y_validation, y_val_pred, y_test, y_test_pred)

#Support Vector Machine
clf = models.SVM(X_train, y_train)
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)
y_val_pred = clf.predict(X_validation)
y_test_pred = clf.predict(X_test)
print_scores("SUPPORT VECTOR MACHINE", y_validation, y_val_pred, y_test, y_test_pred)

sys.stdout = prev_stdout
f.close()