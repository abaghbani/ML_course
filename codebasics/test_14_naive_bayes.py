import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine

wine = load_wine()
print(wine.keys())
# print(wine.DESCR)
# print(wine.data[:5])
print(wine.feature_names)
print(wine.target_names)

y = wine.target
X = wine.data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)
print(y_pred[:10],y_test[:10])

def exercise_2():
    df = pd.read_csv("test_14_spam.csv")
    df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
    df.drop(['Category'],axis='columns',inplace=True)
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df['Message'],df['spam'],test_size=0.2)

    # cv = CountVectorizer()
    # X_train_cv = cv.fit_transform(X_train)
    # print(X_train_cv.toarray()[:3])
    # model = MultinomialNB()
    # model.fit(X_train_cv.toarray(),y_train)
    # X_test_cv = cv.transform(X_test)
    # print(model.score(X_test_cv.toarray(),y_test))

    emails = [
        'Hey mohan, can we get together to watch footbal game tomorrow?',
        'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
        'Please call me back now',
        'Congratulations! You have won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.',
        'Hey, are we still meeting for dinner tonight?'
    ]
    # emails_count = cl.transform(emails)
    # print(model.predict(emails_count))

    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    print(clf.predict(emails))

def exercise1():
    df = pd.read_csv("test_14_titanic.csv")
    df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
    # print(df.head(20))

    inputs = df.drop('Survived',axis='columns')
    target = df.Survived
    inputs.Age = inputs.Age.fillna(inputs.Age.mean())
    inputs.Sex = inputs.Sex.map({'male':0,'female':1})
    # dummies = pd.get_dummies(inputs.Sex)
    # inputs = pd.concat([inputs,dummies],axis='columns')
    # inputs.drop('Sex',axis='columns',inplace=True)
    print(inputs.head())

    X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
    model = GaussianNB()
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))
    print(cross_val_score(GaussianNB(),X_train, y_train, cv=5))

