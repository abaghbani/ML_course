import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('test_01_canada_per_capita_income.csv')
print(df.head(5))

x_train = df.iloc[:,0].values.reshape(-1,1)
x_train = np.column_stack((x_train, x_train**2))
y_train = df.iloc[:,1].values.T
print(x_train.shape)
print(y_train.shape)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model.predict([[2020, 2020**2]]))
print(model.coef_)
print(model.intercept_)

plt.scatter(x_train[:, 0], y_train)
x_train = np.vstack((x_train, [2020, 2020**2]))
print(x_train.shape)
plt.plot(x_train[:, 0], model.predict(x_train))
plt.grid()
plt.show()