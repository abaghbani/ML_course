import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.model_selection as mdl
import sklearn.preprocessing as prep
import sklearn.linear_model as lmdl
import sklearn.datasets as ds
import sklearn.metrics as metr
import sklearn.tree as sk_tree
import sklearn.utils as sk_utils
import sklearn.ensemble as sk_ensemble

class DecisionStump:

    # A decision stump classifier for multi-class classification problems (depth = 1).
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.value_left = None
        self.value_right = None

    def fit(self, X, y):
        # Fits a decision stump to the dataset (X, y).
        best_gain = -1
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                if len(left_y) and len(right_y):
                    left_weight = len(left_y) / len(y)
                    right_weight = 1 - left_weight
                    gain = self._entropy(y) - (left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y))
                    if gain > best_gain:
                        best_gain = gain
                        self.feature = feature_index
                        self.threshold = threshold
                        self.value_left = np.bincount(left_y).argmax()
                        self.value_right = np.bincount(right_y).argmax()

    def predict(self, X):
        # Predicts class labels for samples in X.
        return np.where(X[:, self.feature] <= self.threshold, self.value_left, self.value_right)

    def _entropy(self, y):
        # Computes entropy for a set of labels.
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

def test1():
    iris = ds.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y.flatten(), test_size=0.2, random_state=42)
    stump = DecisionStump()
    stump.fit(X_train, y_train)
    stump_predictions = stump.predict(X_test)

    print(f"Iris features: {iris.feature_names}")
    print(f"Iris target: {iris.target_names}")

    print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    print(f"Decision Stump Accuracy: {metr.accuracy_score(y_test, stump_predictions):.3f}")
    print(f"Decision Stump F1-Score: {metr.f1_score(y_test, stump_predictions, average='weighted'):.3f}")

    ## another decision tree approach:
    dt_sklearn = sk_tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    dt_sklearn.fit(X_train, y_train)
    dt_skl_predictions = dt_sklearn.predict(X_test)

    print(f"Sklearn DT Accuracy: {metr.accuracy_score(y_test, dt_skl_predictions):.3f}")
    print(f"Sklearn DT F1-Score: {metr.f1_score(y_test, dt_skl_predictions, average='weighted'):.3f}")

    plt.figure(figsize=(10,6))
    plt.title("Decision Tree Visualization - Sklearn")
    sk_tree.plot_tree(dt_sklearn, feature_names=list(iris.feature_names), class_names=list(iris.target_names), filled=True, rounded=True)
    plt.show()

class RandomForest:
    
    # A random forest classifier for multi-class classification problems (using decision stumps with depth 1).
    def __init__(self, n_trees=7):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        # Fits a random forest to the dataset (X, y).
        self.trees = []
        for _ in range(self.n_trees):
            stump = DecisionStump()
            X_sample, y_sample = self._bootstrap_samples(X, y)
            stump.fit(X_sample, y_sample)
            self.trees.append(stump)

    def predict(self, X):
        # Predicts class labels for samples in X.
        stump_predictions = np.array([stump.predict(X) for stump in self.trees])
        return self._majority_vote(stump_predictions)
    
    def _bootstrap_samples(self, X, y):
        # Applies bootstrap resampling to the dataset.
        return sk_utils.resample(X, y, n_samples=len(X), replace=True)
    
    def _majority_vote(self, predictions):
        # Returns the majority vote of the predictions.
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def test2():
    breast_cancer = ds.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y, test_size=0.2, random_state=42)
    rf_custom = RandomForest()
    rf_custom.fit(X_train, y_train)
    rf_cust_predictions = rf_custom.predict(X_test)

    print(f"Breast Cancer features: {breast_cancer.feature_names}")
    print(f"Breast Cancer target: {breast_cancer.target_names}")

    print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")


    print(f"Custom RF Accuracy: {metr.accuracy_score(y_test, rf_cust_predictions):.3f}")
    print(f"Custom RF F1-Score: {metr.f1_score(y_test, rf_cust_predictions, average='weighted'):.3f}")


    rf_sklearn = sk_ensemble.RandomForestClassifier(n_estimators=7, max_depth=1, criterion='entropy', random_state=42)
    rf_sklearn.fit(X_train, y_train.ravel())
    rf_skl_predictions = rf_sklearn.predict(X_test)

    print(f"Sklearn RF Accuracy: {metr.accuracy_score(y_test, rf_skl_predictions):.3f}")
    print(f"Sklearn RF F1-Score: {metr.f1_score(y_test, rf_skl_predictions, average='weighted'):.3f}")

    for idx, tree in enumerate(rf_sklearn.estimators_):
        plt.figure(figsize=(8,6))
        sk_tree.plot_tree(tree, filled=True, feature_names=list(breast_cancer.feature_names), class_names=list(breast_cancer.target_names))
        plt.title(f"Random Forest Visualization - Tree {idx + 1}")
        plt.tight_layout()
        plt.show()

    sample_idx = 112

    pd.set_option('display.max_columns', None)
    pd.DataFrame(X_test[112].reshape(1, -1), columns=breast_cancer.feature_names).head()

    votes = [tree.predict(X_test[112].reshape(1, -1)) for tree in rf_sklearn.estimators_]
    final_prediction = rf_sklearn.predict(X_test[sample_idx].reshape(1, -1))[0]

    plt.figure(figsize=(8, 4))
    plt.scatter([range(1, len(rf_sklearn.estimators_) + 1)], votes, s=100, alpha=0.7, label='Votes')
    plt.axhline(y=final_prediction, color='r', linestyle='--', label='Final Prediction')
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    plt.xlabel('Decision Trees')
    plt.ylabel('Votes')
    plt.title(f'Random Forest: Votes from Each DT for Sample #{sample_idx + 1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

test2()

def test3():
    breast_cancer = ds.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y, test_size=0.2, random_state=42)
    ## XGBoost:
    # import xgboost as xgb
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train.ravel())

    xgb_predictions = xgb_model.predict(X_test)

    print(f"XGB Accuracy: {metr.accuracy_score(y_test, xgb_predictions):.3f}")
    print(f"XGB F1-Score: {metr.f1_score(y_test, xgb_predictions, average='weighted'):.3f}")

    xgb_model.get_booster().feature_names = list(breast_cancer.feature_names)
    # graph = xgb.to_graphviz(xgb_model)
    # graph

## comparison:
def test_dataset(X, y):
    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = sk_ensemble.RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    rf_accuracy = metr.accuracy_score(y_test, rf_predictions)
    xgb_accuracy = metr.accuracy_score(y_test, xgb_predictions)

    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    cm_rf = metr.confusion_matrix(y_test, rf_predictions)
    metr.ConfusionMatrixDisplay(cm_rf, display_labels=list(breast_cancer.target_names)).plot(ax=ax[0])
    ax[0].set_title('Random Forest Confusion Matrix')

    cm_xgb = metr.confusion_matrix(y_test, xgb_predictions)
    metr.ConfusionMatrixDisplay(cm_xgb, display_labels=list(breast_cancer.target_names)).plot(ax=ax[1])
    ax[1].set_title('XGBoost Confusion Matrix')

    plt.show()

    print("\nRandom Forest Classification Report:\n", metr.classification_report(y_test, rf_predictions, target_names=list(breast_cancer.target_names)))
    print("-" * 60)
    print("\nXGBoost Classification Report:\n", metr.classification_report(y_test, xgb_predictions, target_names=list(breast_cancer.target_names)))

def test4():
    breast_cancer = ds.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    test_dataset(X, y)

    pd1 = pd.read_csv("./Introduction_to_Machine_Learning-main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble Learning/assets/imbalanced_datasets/1.csv")
    pd2 = pd.read_csv("./Introduction_to_Machine_Learning-main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble Learning/assets/imbalanced_datasets/2.csv")
    pd3 = pd.read_csv("./Introduction_to_Machine_Learning-main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble Learning/assets/imbalanced_datasets/3.csv")

    print(f"Shape of the first dataset: {pd1.shape}")
    print(f"Shape of the second dataset: {pd2.shape}")
    print(f"Shape of the third dataset: {pd3.shape}")

    X1, y1 = pd1.drop(columns=['target']), pd1['target']
    X2, y2 = pd2.drop(columns=['target']), pd2['target']
    X3, y3 = pd3.drop(columns=['target']), pd3['target']

    test_dataset(X1, y1)

    test_dataset(X2, y2)

    test_dataset(X3, y3)