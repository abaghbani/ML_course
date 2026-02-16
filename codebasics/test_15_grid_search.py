import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def exercise_5():
    from sklearn.datasets import load_digits
    digit = load_digits()

    model_params = {
        'svm': {
            'model': SVC(gamma='auto'),
            'params' : {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }  
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [1,5,10]
            }
        },
        'logistic_regression' : {
            'model': LogisticRegression(solver='liblinear',multi_class='auto'),
            'params': {
                'C': [1,5,10]
            }
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params': {}
        },
        'multinomial_nb': {
            'model': MultinomialNB(),
            'params': {}
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini','entropy'],
                'splitter': ['best','random']
            }
        }
    }

    for model_name, mp in model_params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(digit.data, digit.target)
        logger.info(model_name)
        logger.info("Best score: %s", clf.best_score_)
        logger.info("Best params: %s", clf.best_params_)

def exercise_4():
    from sklearn.datasets import load_iris
    iris = load_iris()

    model_params = {
        'svm': {
            'model': SVC(gamma='auto'),
            'params' : {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }  
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [1,5,10]
            }
        },
        'logistic_regression' : {
            'model': LogisticRegression(solver='liblinear',multi_class='auto'),
            'params': {
                'C': [1,5,10]
            }
        }
    }

    scores = []

    for model_name, mp in model_params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(iris.data, iris.target)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        
    logger.info("Grid Search Results: %s", pd.DataFrame(scores,columns=['model','best_score','best_params']))

def exercise_3():
    from sklearn.datasets import load_iris
    iris = load_iris()

    rs = RandomizedSearchCV(SVC(gamma='auto'), {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }, 
        cv=5, 
        return_train_score=False, 
        n_iter=2
    )
    rs.fit(iris.data, iris.target)
    result = rs.cv_results_
    logger.info("Randomized Search Results: %s", pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']])

def exercise_2():
    from sklearn.datasets import load_iris
    iris = load_iris()

    clf = GridSearchCV(SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    
    logger.info("CV Results: %s", clf.cv_results_)
    logger.info("Params: %s", clf.cv_results_['params'])
    logger.info("Best estimator: %s", clf.best_estimator_)
    logger.info("Best score: %s", clf.best_score_)

def exercise_1():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2)
    model = SVC(kernel='rbf', C=10.0, gamma='scale')
    model.fit(X_train, y_train)
    logger.info("Model accuracy: %s", model.score(X_test, y_test))

    score = cross_val_score(SVC(kernel='rbf', C=10.0, gamma='scale'), iris.data, iris.target, cv=5)
    logger.info("Cross validation scores: %s", score)
    logger.info("Average cross validation score: %s", score.mean())

    score = cross_val_score(SVC(kernel='linear', C=10.0, gamma='scale'), iris.data, iris.target, cv=5)
    logger.info("Cross validation scores: %s", score)
    logger.info("Average cross validation score: %s", score.mean())

    score = cross_val_score(SVC(kernel='rbf', C=12.0, gamma='scale'), iris.data, iris.target, cv=5)
    logger.info("Cross validation scores: %s", score)
    logger.info("Average cross validation score: %s", score.mean())

if __name__ == "__main__":
    logging.basicConfig(filename='test_15_grid_search.log', filemode='a', format='%(filename)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(
        description='Run exercises for grid search',
        epilog='Example usage: python test_15_grid_search.py 1',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    argparser.add_argument('--exercise', type=int, default=1, help='Exercise number to run')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = argparser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    action = {
        1: exercise_1,
        2: exercise_2,
        3: exercise_3,
        4: exercise_4,
        5: exercise_5
    }
    if args.exercise in action:
        action[args.exercise]()
    else:
        logger.warning(f"Exercise {args.exercise} not defined")