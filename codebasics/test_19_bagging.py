import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

def exercise_1(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset description: {df.describe()}")
    logger.info(f"Feature names: {df.columns.tolist()}")
    logger.info(f"First 5 rows :\n{df.head()}")
    logger.info(f"target variable distribution:\n{df['Outcome'].value_counts()}")
    logger.info(f'{df.isnull().sum()}')

    X = df.drop('Outcome', axis='columns')
    y = df.Outcome

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
    logger.info(f'score values of DT: {scores.mean()}')

    from sklearn.ensemble import BaggingClassifier
    bag_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        oob_score=True,
        random_state=0
    )
    bag_model.fit(X_train, y_train)
    logger.info(f'bagging oob score is: {bag_model.oob_score_}')
    logger.info(f'bagging score is: {bag_model.score(X_test, y_test)}')

    bag_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        oob_score=True,
        random_state=0
    )
    scores = cross_val_score(bag_model, X, y, cv=5)
    logger.info(f'bagging of DT, cross val score is: {scores.mean()}')

    from sklearn.ensemble import RandomForestClassifier
    scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
    logger.info(f'bagging of RF, cross val score is: {scores.mean()}')

def exercise_2(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset description: {df.describe()}")
    logger.info(f"Feature names: {df.columns.tolist()}")
    logger.info(f"First 5 rows :\n{df.head()}")
    # logger.info(f'{df.isnull().sum()}')
    # logger.info(f"Sex value:\n{df['Sex'].unique()}")
    # logger.info(f"ChestPainType value:\n{df['ChestPainType'].unique()}")
    # logger.info(f"RestingECG value:\n{df['RestingECG'].unique()}")
    # logger.info(f"ExerciseAngina value:\n{df['ExerciseAngina'].unique()}")
    # logger.info(f"ST_Slope value:\n{df['ST_Slope'].unique()}")

    df = df[df['Cholesterol'] <= (df['Cholesterol'].mean() + 3*df['Cholesterol'].std())]
    df = df[df['Oldpeak'] <= (df['Oldpeak'].mean() + 3*df['Oldpeak'].std())]
    df = df[df['RestingBP'] <= (df['RestingBP'].mean() + 3*df['RestingBP'].std())].copy()

    logger.info(f"Dataset shape: {df.shape}")

    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 1, 'Flat': 2, 'Down': 3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 1, 'ST': 2, 'LVH': 3})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('HeartDisease', axis='columns').values
    y = df.HeartDisease

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)
    logger.info(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')

    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(SVC(), X, y, cv=5)
    logger.info(f'score svc: {scores.mean()}')

    from sklearn.ensemble import BaggingClassifier
    bag_model = BaggingClassifier(estimator=SVC(), n_estimators=100, max_samples=0.8, random_state=0)
    scores = cross_val_score(bag_model, X, y, cv=5)
    logger.info(f'score bagging: {scores.mean()}')

    from sklearn.tree import DecisionTreeClassifier
    scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5)
    logger.info(f'score DT: {scores.mean()}')
    bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=100, max_samples=0.9, oob_score=True, random_state=0)
    scores = cross_val_score(bag_model, X, y, cv=5)
    logger.info(f'score bagging DT: {scores.mean()}')

    from sklearn.ensemble import RandomForestClassifier
    scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
    logger.info(f'score RF: {scores.mean()}')
    bag_model = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=100, max_samples=0.9, oob_score=True, random_state=0)
    scores = cross_val_score(bag_model, X, y, cv=5)
    logger.info(f'score bagging RF: {scores.mean()}')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Run exercises for Bagging',
        epilog='Example usage: python test_19_bagging.py 1',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    argparser.add_argument('--exercise', type=int, default=1, help='Exercise number to run')
    argparser.add_argument('-f', '--file', type=str, default=None, help='Verbose output')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(filename='test_19_bagging.log', filemode='a', level= log_level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    action = {
        1: lambda: exercise_1(args.file),
        2: lambda: exercise_2(args.file),
    }
    if args.exercise in action:
        logger.info(f"Running exercise {args.exercise}")
        action[args.exercise]()
    else:
        logger.warning(f"Exercise {args.exercise} not defined")

    logger.info("=" * 20)
    logger.info("Script execution completed. \n"+"=" * 50)