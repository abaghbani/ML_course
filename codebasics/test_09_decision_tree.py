import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix

df = pd.read_csv('test_09_titanic.csv')
# print(df.head())

x = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

x.Sex = x.Sex.map({'male':1,'female':2})
x.Age = x.Age.fillna(x.Age.mean())
print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

def excercise_1():
    df = pd.read_csv('test_09_salaries.csv')
    # print(df.head())

    x = df.drop(['salary_more_then_100k'],axis='columns')

    y = df['salary_more_then_100k']
    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()
    x['company'] = le_company.fit_transform(x['company'])
    x['job'] = le_job.fit_transform(x['job'])
    x['degree'] = le_degree.fit_transform(x['degree'])
    print(x.head())

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    model = tree.DecisionTreeClassifier()
    model.fit(x,y)
    print(model.score(x_test,y_test))
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    sn.heatmap(cm,annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
