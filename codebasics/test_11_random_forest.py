import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def exercise_2():
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

    model = RandomForestClassifier(n_estimators=20)
    model.fit(x_train, y_train)
    print("Model accuracy:", model.score(x_test, y_test))
    y_predicted = model.predict(x_test)
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=12)
model.fit(x_train, y_train)
print("Model accuracy:", model.score(x_test, y_test))
