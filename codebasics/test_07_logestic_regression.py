import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('test_07_HR_comma_sep.csv')

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
subdf['salary'] = subdf['salary'].map(salary_mapping)
x = subdf
y = df['left']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train, y_train)
print("Model accuracy:", model.score(x_test, y_test))
y_predicted = model.predict(x_test)
print("Predicted values:", y_predicted)
print("Predicted probabilities:", model.predict_proba(x_test))

def exercise_2():
    df = pd.read_csv('test_07_HR_comma_sep.csv')

    pd.crosstab(df.salary,df.left).plot(kind='bar')
    plt.show()

    pd.crosstab(df.Department,df.left).plot(kind='bar')
    plt.show()

    # Visualizations manually (still is not working well)
    salary_items = df['salary'].unique()
    plt.bar(salary_items, [ ((df['left'] == 1) & (df['salary'] == item)).sum() for item in salary_items], label='Left Company by Salary')
    plt.bar(salary_items, [ ((df['left'] == 0) & (df['salary'] == item)).sum() for item in salary_items], label='Stayed Company by Salary')
    plt.xlabel('Salary')
    plt.ylabel('Number of Employees')
    xxx = np.arange(len(salary_items))
    plt.xticks(xxx, salary_items)
    plt.legend()
    plt.show()

    depart_items = df['Department'].unique()
    plt.bar(depart_items, [ ((df['left'] == 1) & (df['Department'] == item)).sum() for item in depart_items])
    plt.xlabel('Department')
    plt.ylabel('Left Company')
    plt.xticks(rotation=45)
    plt.show()

def exercise_1():
    df = pd.read_csv('test_07_insurance_data.csv')

    x = df.age.values.reshape(-1, 1)
    y = df.bought_insurance.values

    # plt.scatter(x, y, marker='*', color='red')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    print("Model accuracy:", model.score(x_test, y_test))
    y_predicted = model.predict(x_test)
    print("Predicted values:", y_predicted)

    print("Predicted probabilities:", model.predict_proba(x_test))
