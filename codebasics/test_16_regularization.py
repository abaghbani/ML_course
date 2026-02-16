import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB

def exercise_1(filename):
    if filename is None:
        logger.error("Filename is required for exercise 1")
        return
    dataset = pd.read_csv(filename)
    logger.debug(f"Dataset shape: {dataset.shape}")
    logger.debug(f"Dataset columns: {dataset.columns}")
    logger.debug(f"Dataset head:\n{dataset.head()}")

    # let's use limited columns which makes more sense for serving our purpose
    cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
                'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
    dataset = dataset[cols_to_use]
    logger.debug(f"Number of Nan values in each column:\n{dataset.isna().sum()}")
    
    ########## Data preprocessing
    # Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
    # like 0 for Propertycount, Bedroom2 will refer to other class of NA values
    # like 0 for Car feature will mean that there's no car parking feature with house
    cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
    dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
    # other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
    # using Lasso and Ridge Regression
    dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
    dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
    dataset.dropna(inplace=True)
    dataset = pd.get_dummies(dataset, drop_first=True)
    X = dataset.drop('Price', axis=1)
    y = dataset['Price']
    
    from sklearn import model_selection
    train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y, test_size=0.3, random_state=2)
    from sklearn import linear_model
    reg = linear_model.LinearRegression().fit(train_X, train_y)
    logger.info(f"R2 score on test set: {reg.score(test_X, test_y)}")

    lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
    lasso_reg.fit(train_X, train_y)
    logger.info(f"R2 score on test set (Lasso): {lasso_reg.score(test_X, test_y)}")

    ridge_reg= linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
    ridge_reg.fit(train_X, train_y)
    logger.info(f"R2 score on test set (Ridge): {ridge_reg.score(test_X, test_y)}")

def exercise_2():
    pass

if __name__ == "__main__":
    logging.basicConfig(filename='test_16_regularization.log', filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(
        description='Run exercises for regularization',
        epilog='Example usage: python test_16_regularization.py 1',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    argparser.add_argument('--exercise', type=int, default=1, help='Exercise number to run')
    argparser.add_argument('-f', '--file', type=str, default=None, help='Verbose output')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = argparser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    action = {
        1: lambda: exercise_1(args.file),
        2: exercise_2
    }
    if args.exercise in action:
        logger.info(f"Running exercise {args.exercise}")
        action[args.exercise]()
    else:
        logger.warning(f"Exercise {args.exercise} not defined")

    logger.info("Script execution completed")
    logger.info("\n"+"=" * 50)