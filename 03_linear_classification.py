import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import sklearn.model_selection as mdl
import sklearn.preprocessing as prep
import sklearn.linear_model as lmdl
import sklearn.datasets as ds
import sklearn.metrics as metr
import pandas as pd

def section1():
    np.random.seed(42)
    class_A = np.random.normal(loc=(4, 4), scale=0.25, size=(100, 2))
    labels_A = np.ones(class_A.shape[0])

    class_B = np.random.normal(loc=(-1, -1), scale=1.5, size=(50, 2))
    labels_B = np.zeros(class_B.shape[0])

    X = np.vstack((class_A, class_B))
    y = np.hstack((labels_A, labels_B))

    model = lmdl.LinearRegression()
    model.fit(X, y)

    w1, w2 = model.coef_  # weights
    w0 = model.intercept_  # bias

    print(f"Model weights: w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {w0:.2f}")

    # Plot the decision boundary: w1 * x1 + w2 * x2 + b = 0.5
    x_vals = np.linspace(-4, 6, 100)
    decision_boundary = (-w1 * x_vals - w0 + 0.5) / w2

    plt.figure(figsize=(8, 6))
    plt.scatter(class_A[:, 0], class_A[:, 1], label='Class A', color='blue')
    plt.scatter(class_B[:, 0], class_B[:, 1], label='Class B', color='red')
    plt.plot(x_vals, decision_boundary, label='Decision Boundary (SSE)', color='green')
    plt.title("Linear Classifier Using SSE for Classification")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

def section2():
    def generate_ab_class(n_points=100):
        X = np.random.uniform(0, 10, n_points)
        Y = np.random.uniform(0, 10, n_points)
        class_A = np.array([[x, y] for x,y in zip(X,Y) if x<y])
        class_B = np.array([[x, y] for x,y in zip(X,Y) if x>y])
        return class_A, class_B

    def generate_xor_data(n_points=200, seed=42):
        np.random.seed(seed)
        X = np.random.uniform(0, 1, n_points)
        Y = np.random.uniform(0, 1, n_points)
        class_A = np.array([[x, y] for x,y in zip(X,Y) if (x > 0.5 and y > 0.5) or (x < 0.5 and y < 0.5)])
        class_B = np.array([[x, y] for x,y in zip(X,Y) if not((x > 0.5 and y > 0.5) or (x < 0.5 and y < 0.5))])
        return class_A, class_B

    def generate_data(n_points=200, seed=42):
        np.random.seed(seed)

        # Class 0: points inside a circle with radius 5
        radius_0 = 5
        theta_0 = np.random.uniform(0, 2 * np.pi, n_points)
        r_0 = radius_0 * np.sqrt(np.random.uniform(0, 1, n_points))  # sqrt for uniform distribution
        x0 = r_0 * np.cos(theta_0)
        y0 = r_0 * np.sin(theta_0)
        class_0 = np.vstack((x0, y0)).T

        # Class 1: points in an annulus between radius 8 and 10
        inner_radius_1 = 8
        outer_radius_1 = 10
        theta_1 = np.random.uniform(0, 2 * np.pi, n_points)
        r_1 = np.sqrt(np.random.uniform(inner_radius_1**2, outer_radius_1**2, n_points))
        x1 = r_1 * np.cos(theta_1)
        y1 = r_1 * np.sin(theta_1)
        class_1 = np.vstack((x1, y1)).T

        return class_0, class_1

    class_A, class_B = generate_data()
    # class_A, class_B = generate_xor_data()
    # class_A, class_B = generate_ab_class()

    # plt.figure(figsize=(1, 1))
    plt.scatter(class_A[:, 0], class_A[:, 1], color='green', label='Class A (y > x)')
    plt.scatter(class_B[:, 0], class_B[:, 1], color='orange', label='Class B (y < x)')
    # plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Decision Boundary (y = x)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Classification Based on y > x and y < x')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = []  # storing the number of misclassifications in each epoch

    def fit(self, X, y):
        """
        Train the Perceptron model on the provided data.

        Parameters:
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values. Must be +1 or -1.
        """
        n_samples, n_features = X.shape
        # starting weights and bias equal zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias  # w^T x + b
                y_pred = self._unit_step(linear_output)
                if y[idx] != y_pred: # misclassfied
                    update = self.learning_rate * y[idx]
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)
            # if no errors, convergence achieved
            if errors == 0:
                print(f"Converged after {epoch+1} epochs")
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : array-like, shape = [n_samples, n_features]

        Returns:
        array, shape = [n_samples]
            Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step(linear_output)

    def _unit_step(self, x):
        return np.where(x >= 0, 1, -1)

def section3():
    def generate_ab_class(n_points=100):
        X = np.random.uniform(0, 10, n_points)
        Y = np.random.uniform(0, 10, n_points)
        class_A = np.array([[x, y] for x,y in zip(X,Y) if x<y])
        class_B = np.array([[x, y] for x,y in zip(X,Y) if x>y])
        return class_A, class_B

    class_A, class_B = generate_ab_class()

    X_ab = np.vstack((class_A, class_B))
    y_ab = np.hstack((np.ones(class_A.shape[0]), -np.ones(class_B.shape[0])))

    shuffle_idx = np.random.permutation(len(X_ab))
    X_ab, y_ab = X_ab[shuffle_idx], y_ab[shuffle_idx]

    print("Combined Data Sample Points:\n", X_ab[:5])
    print("Combined Labels:\n", y_ab[:5])

    perceptron = Perceptron(learning_rate=0.01, n_epochs=1000)
    perceptron.fit(X_ab, y_ab)

    print(f"Final Weights: {perceptron.weights}")
    print(f"Final Bias: {perceptron.bias}")

    x_min, x_max = X_ab[:, 0].min() - 1, X_ab[:, 0].max() + 1
    y_min, y_max = X_ab[:, 1].min() - 1, X_ab[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid)
    Z = Z.reshape(xx.shape)

    cmap_light = col.ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = col.ListedColormap(['#FF0000', '#0000FF'])
    plt.figure(figsize=(10, 6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    if perceptron.weights[1] != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    else:
        x_val = -perceptron.bias / perceptron.weights[0]
        plt.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')

    plt.scatter(class_A[:, 0], class_A[:, 1], color='red', marker='o', label='Class A')
    plt.scatter(class_B[:, 0], class_B[:, 1], color='blue', marker='s', label='Class B')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary and Decision Regions')
    plt.legend()
    plt.grid(True)
    plt.show()

def section4():
    data = ds.load_breast_cancer()
    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y

    selected_features = ['mean radius', 'mean texture']
    X_selected = df[selected_features].values
    y_selected = y  # 0 = malignant, 1 = benign


    X_train, X_test, y_train, y_test = mdl.train_test_split(
        X_selected, y_selected, test_size=0.2, random_state=42, stratify=y_selected
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")# convert labels: 0 -> -1, 1 -> 1

    y_train_perceptron = np.where(y_train == 0, -1, 1)
    y_test_perceptron = np.where(y_test == 0, -1, 1)

    perceptron = Perceptron(learning_rate=0.01, n_epochs=1000)

    perceptron.fit(X_train, y_train_perceptron)

    print(f"Final Weights: {perceptron.weights}")
    print(f"Final Bias: {perceptron.bias}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(perceptron.errors_) + 1, 20), perceptron.errors_[::20], marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Misclassifications')
    plt.title('Perceptron Learning Progress')
    plt.grid(True)
    plt.show()

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    plt.figure(figsize=(10, 6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    if perceptron.weights[1] != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    else:
        x_val = -perceptron.bias / perceptron.weights[0]
        plt.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')

    # Malignant: 0 (red), Benign: 1 (blue)
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='red', marker='o', edgecolor='k', label='Malignant')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='blue', marker='s', edgecolor='k', label='Benign')

    plt.ylim(10,40)
    plt.xlabel('Mean Radius')
    plt.ylabel('Mean Texture')
    plt.title('Perceptron Decision Boundary and Decision Regions (Breast Cancer Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()
    