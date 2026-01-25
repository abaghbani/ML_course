import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('test_02_hiring.csv')

word_to_int = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12
}

df.experience = df.experience.map(word_to_int)
median_test_score = df['test_score(out of 10)'].mean()
print(median_test_score)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(int(median_test_score))
df = df.fillna(0)
print(df)

x_train = df.iloc[:,:3].values.reshape(-1,3)
y_train = df.iloc[:,3].values.T
print(x_train.shape)
print(y_train.shape)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.predict([[2,9,6]]))
print(model.predict([[12,10,10]]))

print(model.coef_)
print(model.intercept_)

joblib.dump(model, 'my_model')

plt.scatter(x_train[:, 0], y_train)
x_train = np.vstack((x_train, [2020, 2020**2]))
print(x_train.shape)
plt.plot(x_train[:, 0], model.predict(x_train))
plt.grid()
plt.show()

