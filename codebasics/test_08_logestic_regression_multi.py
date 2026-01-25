import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
print("Iris feature names:", iris.feature_names)
print("Iris target names:", iris.target_names)
print(iris.data[0:5])
print(iris.target[0:5])

x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)
print("Model accuracy:", model.score(x_test, y_test))
y_predicted = model.predict(x_test)
print("Predicted values:", y_predicted)
print("Predicted probabilities:", model.predict_proba(x_test))

cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)


def exercise_1():
    from sklearn.datasets import load_digits
    digits = load_digits()

    # plt.gray() 
    # for i in range(5):
    #     plt.matshow(digits.images[i]) 

    # print("Digit labels:", digits.target)
    # print(dir(digits))

    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("Model accuracy:", model.score(x_test, y_test))
    y_predicted = model.predict(x_test)
    # print("Predicted values:", y_predicted)
    # print("Predicted probabilities:", model.predict_proba(x_test))

    cm = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()