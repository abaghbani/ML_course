import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def exercise_1():
    from sklearn.datasets import load_digits
    digits = load_digits()
    
    lr_scores = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)
    print("Logistic Regression scores:", lr_scores)
    svm_scores = cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)
    print("SVM scores:", svm_scores)
    rf_scores = cross_val_score(RandomForestClassifier(n_estimators=60), digits.data, digits.target,cv=3)
    print("Random Forest scores:", rf_scores)

from sklearn.datasets import load_iris
iris = load_iris()

lr_scores = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), iris.data, iris.target,cv=3)
print("Logistic Regression scores:", lr_scores)
svm_scores = cross_val_score(SVC(gamma='auto'), iris.data, iris.target,cv=3)
print("SVM scores:", svm_scores)
rf_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target,cv=3)
print("Random Forest scores:", rf_scores)
dt_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target,cv=3)
print("Decision Tree scores:", dt_scores)