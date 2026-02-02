import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.data[:10])

# scaler = MinMaxScaler()
# scaler.fit(iris.data)
# iris.data = scaler.transform(iris.data)
# print(iris.data[:10])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(iris.data[:, 2:4])
print(y_predicted)
print(km.cluster_centers_)

        
# plt.scatter(iris.data[:,0], iris.data[:,1], c=y_predicted, cmap='rainbow')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.show()

plt.scatter(iris.data[:,2], iris.data[:,3], c=y_predicted, cmap='rainbow')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(iris.data[:, 2:4])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()


def exercise_1():
    df = pd.read_csv('test_13_income.csv')

    scaler = MinMaxScaler()
    scaler.fit(df[['Income($)']])
    df['Income($)'] = scaler.transform(df[['Income($)']])
    scaler.fit(df[['Age']])
    df['Age'] = scaler.transform(df[['Age']])

    print(df.head())

    plt.scatter(df.Age,df['Income($)'])
    plt.xlabel('Age')
    plt.ylabel('Income($)')
    plt.show()

    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(df[['Age','Income($)']])
    print(y_predicted)
    print(km.cluster_centers_)


    df['cluster']=y_predicted
    print(df.head())
    plt.scatter(df.Age,df['Income($)'],c=df.cluster, cmap='rainbow')
    plt.xlabel('Age')
    plt.ylabel('Income($)')
    plt.show()

    sse = []
    k_rng = range(1,10)
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(df[['Age','Income($)']])
        sse.append(km.inertia_)

    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(k_rng,sse)
    plt.show()
