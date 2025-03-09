import numpy as np
import matplotlib.pyplot as plt

def generate_data(n=50, noise=5.0):
    np.random.seed(42)
    X = np.linspace(-10, 10, n)
    # Ground truth line: y = 3x + 8
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    y = true_slope * X + true_intercept + noise
    return X, y


# Hypothesis: h_w(x) = w_0 + w_1 * x_1
def h_w(x, w):
    return w[0] + w[1] * x  # equivalent to w_0 + w_1 * x

# Linear Regression using closed-form solution
def linear_regression_closed_form(X, y):
    # Adding bias term (x_0 = 1) to input vector X
    X_b = np.c_[np.ones((len(X), 1)), X]  # X_b is now the full input vector with bias term
    # Closed-form solution: w = (X^T * X)^-1 * X^T * y
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return w

# regression analytic
def test1():
    X, y = generate_data(n=50, noise=5.0)
    # plt.scatter(X, y, color='blue', label='Data Points')
    # plt.title("Generated Data (Univariate)")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.legend()
    # plt.show()

    # Get parameter vector w
    w = linear_regression_closed_form(X, y)
    print(f"Parameters (w): ")
    print(f"w_1 = {w[1]:.2f}, w_0 = {w[0]:.2f}")

    y_pred = h_w(X, w)

    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Prediction (Closed Form)')
    plt.title("Linear Regression - Closed Form Solution")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

# Function to generate polynomial features (input matrix X')
def polynomial_features(X, degree):
    X_poly = np.c_[np.ones(len(X))]
    for i in range(1, degree + 1):
        X_poly = np.c_[X_poly, X**i]
    return X_poly

def polynomial_regression(X, y, degree):
    X_poly = polynomial_features(X, degree)
    # Closed-form solution: w = (X'^T * X')^-1 * X'^T * y
    w = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return w

# regression polynomial
def test2():
    X, y = generate_data(n=50, noise=5.0)

    m = 5  # Degree of the polynomial regression
    w_poly = polynomial_regression(X, y, m)  # Parameter vector w

    print(f"Parameters (w) for Degree {m}: {w_poly}")

    X_fit = np.linspace(X.min(), X.max(), 200)
    X_fit_poly = polynomial_features(X_fit, m)
    y_poly_pred = X_fit_poly.dot(w_poly)  # h_w(x) = X' * w

    # Plot the actual data and the polynomial fit
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_fit, y_poly_pred, color='green', label=f'Polynomial Fit (Degree {m})')
    plt.title(f"Polynomial Regression (Degree {m})")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

def compute_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# regression over train and test dataset
def test3():
    from sklearn.model_selection import train_test_split

    X, y = generate_data(n=50, noise=5.0)

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degrees = range(0, 9)
    train_rms_errors = []
    test_rms_errors = []


    for d in degrees:
        # Train the model on the training set
        w_poly = polynomial_regression(X_train, y_train, d)

        # Compute predictions for the training set
        X_train_poly = polynomial_features(X_train, d)
        y_train_pred = X_train_poly.dot(w_poly)

        # Compute predictions for the test set
        X_test_poly = polynomial_features(X_test, d)
        y_test_pred = X_test_poly.dot(w_poly)

        # Calculate RMSE for both training and test sets
        train_rms_error = compute_rms_error(y_train, y_train_pred)
        test_rms_error = compute_rms_error(y_test, y_test_pred)

        # Store the errors
        train_rms_errors.append(train_rms_error)
        test_rms_errors.append(test_rms_error)

        # Print the RMSE for the current degree
        print(f"Degree {d}: Train RMSE = {train_rms_error:.2f}, Test RMSE = {test_rms_error:.2f}")

        # Plot the polynomial fit on the training data
        plt.scatter(X_train, y_train, color='blue', label="Training Data")
        plt.scatter(X_test, y_test, color='red', label="Test Data", alpha=0.6)
        X_fit = np.linspace(X.min(), X.max(), 200)
        X_fit_poly = polynomial_features(X_fit, d)
        y_fit_pred = X_fit_poly.dot(w_poly)
        plt.plot(X_fit, y_fit_pred, label=f"Degree {d} Fit", color='green')
        plt.title(f"Polynomial Regression (Degree {d})")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.show()

    # Plot RMSE for training and test sets
    plt.plot(degrees, train_rms_errors, marker='o', linestyle='-', color='blue', label='Train RMSE')
    plt.plot(degrees, test_rms_errors, marker='o', linestyle='-', color='red', label='Test RMSE')
    plt.title("Train vs Test RMSE vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.xticks(degrees)
    plt.grid(True)
    plt.legend()
    plt.show()

# SSE cost function
def cost_function(X, y, w):
    return np.sum((h_w(X, w) - y)**2) / len(X)

# Gradient descent
def gradient_descent(X, y, w, alpha, num_iters):
    m = len(X)
    cost_history = []
    w_history = [w.copy()]

    for i in range(num_iters):
        # updates
        gradient_w0 = np.sum(h_w(X, w) - y)
        gradient_w1 = np.sum((h_w(X, w) - y) * X)
        w[0] -= alpha * gradient_w0
        w[1] -= alpha * gradient_w1

        cost_history.append(cost_function(X, y, w))
        w_history.append(w.copy())  # Store a copy of w, not the reference

    return w, cost_history, w_history

# regression with gradiant descent
def test4():
    X, y = generate_data(n=50, noise=5.0)
    w_initial = [0, 0]  # Start with w0 = 0, w1 = 0
    eta = 0.001  # Learning rate
    num_iters = 500

    # Run Gradient Descent
    w_final, cost_history, w_history = gradient_descent(X, y, w_initial, eta, num_iters)

    # Plot GD Progression (without labels for lines, different alphas)
    plt.scatter(X, y, color='blue', label='Actual Data')

    # Plot lines for every 50th step with increasing alpha
    for idx, w in enumerate(w_history[::num_iters // 100]):
        alpha = 0.15 + 0.85*(idx) / 100  # Gradually increase alpha for each line
        plt.plot(X, h_w(X, w), color='red', alpha=alpha)

    # Final line in bold
    plt.plot(X, h_w(X, w_final), color='red', lw=2, label='Final Line')

    plt.title("GD Progression - Linear Regression")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

    # Visualize cost function (log of J(w))
    # def test5(w_history, cost_history):
    #     X, y = generate_data(n=50, noise=5.0)

    w0_vals = np.linspace(-10, 20, 100)
    w1_vals = np.linspace(-1, 5, 100)
    J_vals = np.zeros((len(w0_vals), len(w1_vals)))

    for i in range(len(w0_vals)):
        for j in range(len(w1_vals)):
            w = [w0_vals[i], w1_vals[j]]
            J_vals[i, j] = cost_function(X, y, w)

    # 3D Plot of J(w)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    # ax.plot_surface(W0, W1, J_vals.T, cmap='viridis')
    ax.plot_surface(W0, W1, np.log(J_vals.T), cmap='viridis', alpha=0.25)
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.set_zlabel('log(J(w))')
    
    # Plot the points on the 3D surface for each GD iteration
    w_history_array = np.array(w_history)  # Convert list to array for easier slicing
    w0_history = w_history_array[:, 0]
    w1_history = w_history_array[:, 1]
    cost_history_log = np.log(np.array(cost_history))  # Log of the cost history

    # Plot the path of gradient descent in 3D
    ax.plot(w0_history[:num_iters], w1_history[:num_iters], cost_history_log, marker='o', color='r', label='GD Path', markersize=3)
    
    plt.title("Cost Function Surface")
    plt.show()

