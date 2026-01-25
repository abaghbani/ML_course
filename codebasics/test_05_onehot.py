import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import compose
import joblib

df = pd.read_csv('test_05_carprices.csv')

le = preprocessing.LabelEncoder()
dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
print(dfle)

X = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
y = dfle.iloc[:,2]
ct = compose.ColumnTransformer([('Car Model', preprocessing.OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:,1:]

print(X)
print(y)

model = linear_model.LinearRegression()
model.fit(X, y)
print(model.score(X, y))
print(model.predict([[0, 1, 45000, 4]]))
print(model.predict([[1, 0, 86000, 7]]))

