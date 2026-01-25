import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

digits = load_digits()

# plt.gray() 
# for i in range(5):
#     plt.matshow(digits.images[i]) 

# print("Digit labels:", digits.target)
# print(dir(digits))

x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = SVC(kernel='rbf', C=10.0, gamma='scale')
model.fit(x_train, y_train)
print("Model accuracy:", model.score(x_test, y_test))


def exercise_1():
    from sklearn.datasets import load_iris
    iris = load_iris()
    print("Iris feature names:", iris.feature_names)
    print("Iris target names:", iris.target_names)
    print(iris.data[0:5])
    print(iris.target[0:5])

    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = SVC(C=10.0)
    model.fit(x_train, y_train)
    print("Model accuracy:", model.score(x_test, y_test))

    print("Prediction for [4.8,3.0,1.5,0.3]:", model.predict([[4.8,3.0,1.5,0.3]]))