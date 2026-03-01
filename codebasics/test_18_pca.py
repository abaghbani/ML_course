import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

def exercise_2(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    logger.info(f"Dataset shape: {df.shape}")
    
    # logger.info(f"Cholesterol min: {df['Cholesterol'].min()}")
    # logger.info(f"Cholesterol max: {df['Cholesterol'].max()}")
    df1 = df[df['Cholesterol'] <= (df['Cholesterol'].mean() + 3*df['Cholesterol'].std())]
    df2 = df1[df1['Oldpeak'] <= (df1['Oldpeak'].mean() + 3*df1['Oldpeak'].std())]
    df3 = df2[df2['RestingBP'] <= (df2['RestingBP'].mean() + 3*df2['RestingBP'].std())].copy()

    # logger.info(f'chest pain types: {df["ChestPainType"].unique()}')
    # df3['ChestPainType_tap'] = df3['ChestPainType'].apply(lambda x: 1 if x == 'TA' else 0)
    # df3['ChestPainType_ata'] = df3['ChestPainType'].apply(lambda x: 1 if x == 'ATA' else 0)
    # df3['ChestPainType_nap'] = df3['ChestPainType'].apply(lambda x: 1 if x == 'NAP' else 0)
    # df3['ChestPainType_asy'] = df3['ChestPainType'].apply(lambda x: 1 if x == 'ASY' else 0)
    # df3.drop('ChestPainType', axis=1, inplace=True)
    # df3['Sex'] = df3['Sex'].map({'M': 1, 'F': 0})
    df3['ExerciseAngina'] = df3['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df3['ST_Slope'] = df3['ST_Slope'].map({'Up': 1, 'Flat': 2, 'Down': 3})
    df3['RestingECG'] = df3['RestingECG'].map({'Normal': 1, 'ST': 2, 'LVH': 3})
    df4 = pd.get_dummies(df3, drop_first=True)

    logger.info(f"Dataset shape after one-hot encoding: {df4.shape}")
    logger.info(f"Feature names after one-hot encoding: {df4.columns.tolist()}")
    logger.info(f"First 5 rows after one-hot encoding:\n{df4.head()}")

    X = df4.drop('HeartDisease', axis=1).values
    y = df4['HeartDisease'].values

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("X_scaled shape: {}".format(X_scaled.shape))
    
    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.2, random_state=30)
    logger.info(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')
    
    from sklearn import ensemble
    model = ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")

    from sklearn import decomposition
    pca = decomposition.PCA(0.95) # Keep 95% variance
    X_pca = pca.fit_transform(X)
    logger.info(f"Original shape: {X.shape}, PCA shape: {X_pca.shape}")
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_pca, y, test_size=0.2, random_state=30)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Random Forest Accuracy after PCA: {accuracy:.4f}")

def exercise_1():
    from sklearn import datasets
    digit = datasets.load_digits()
    
    # plt.matshow(digit.data[9].reshape(8,8))
    # plt.savefig('exercise_2_digit.png')
    # logger.info(f"Dataset shape: {digit.data.shape}")
    # logger.info(f"Feature names: {digit.feature_names}")
    # logger.info(f"Target names: {digit.target_names}")

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(digit.data)
    logger.info("X_scaled shape: {}".format(X_scaled.shape))


    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, digit.target, test_size=0.2, random_state=30)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    from sklearn import linear_model
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")

    from sklearn.decomposition import PCA
    # pca = PCA(0.95) # Keep 95% variance
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digit.data)
    logger.info(f"Original shape: {digit.data.shape}, PCA shape: {X_pca.shape}")

    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_pca, digit.target, test_size=0.2, random_state=30)
    from sklearn import linear_model
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    logging.basicConfig(filename='test_18_pca.log', filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(
        description='Run exercises for PCA',
        epilog='Example usage: python test_18_pca.py 1',
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
        1: exercise_1,
        2: lambda: exercise_2(args.file),
    }
    if args.exercise in action:
        logger.info(f"Running exercise {args.exercise}")
        action[args.exercise]()
    else:
        logger.warning(f"Exercise {args.exercise} not defined")

    logger.info("=" * 20)
    logger.warning("Script execution completed. \n"+"=" * 50)