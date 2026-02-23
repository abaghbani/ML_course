import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def exercise_1(filename):
    # if filename is None:
    #     logger.error("Filename is required for exercise 1")
    #     return

    from sklearn import datasets
    iris = datasets.load_iris()
    
    # logger.info(f"Dataset shape: {iris.data.shape}")
    # logger.info(f"Feature names: {iris.feature_names}")
    # logger.info(f"Target names: {iris.target_names}")

    # plt.scatter(iris.data[:100, 2], iris.data[:100, 3], c=iris.target[:100], cmap='viridis')
    # plt.xlabel(iris.feature_names[0])
    # plt.ylabel(iris.feature_names[1])
    # plt.title('Iris Dataset - KNN Visualization')
    # plt.colorbar()
    # plt.savefig('exercise_1_plot.png')
    # logger.info("Exercise 1 plot saved as 'exercise_1_plot.png'")

    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    logger.info(f"KNN Accuracy: {accuracy:.4f}")

    from sklearn import metrics
    y_pred = knn.predict(X_test)
    report = metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
    logger.info(f"Classification Report:\n{report}")
    cm = metrics.confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")


def exercise_2():
    from sklearn import datasets
    digit = datasets.load_digits()
    
    # logger.info(f"Dataset shape: {digit.data.shape}")
    # logger.info(f"Feature names: {digit.feature_names}")
    # logger.info(f"Target names: {digit.target_names}")

    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(digit.data, digit.target, test_size=0.2, random_state=1)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    logger.info(f"KNN Accuracy: {accuracy:.4f}")

    from sklearn import metrics
    y_pred = knn.predict(X_test)
    report = metrics.classification_report(y_test, y_pred, target_names=digit.target_names.astype(str))
    logger.info(f"Classification Report:\n{report}")
    cm = metrics.confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
if __name__ == "__main__":
    logging.basicConfig(filename='test_17_knn.log', filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(
        description='Run exercises for KNN',
        epilog='Example usage: python test_17_knn.py 1',
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

    logger.info("=" * 20)
    logger.warning("Script execution completed. \n"+"=" * 50)