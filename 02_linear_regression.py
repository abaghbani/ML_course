import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as mdl
import sklearn.preprocessing as prep
import sklearn.linear_model as lmdl
import sklearn.datasets as ds
import sklearn.metrics as metr

# import ssl
# ssl._create_default_https_context = ssl._create_stdlib_context

def generate_data(n=100, noise=10.0):
    np.random.seed(42)
    X = np.random.uniform(-10, 10, n)
    y = X**2 - 2 * X + np.random.randn(n) * noise  # x**2 - 2*x + noise
    return X, y

def compute_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def test1():
    X, y = generate_data(n=15)
    # X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y, test_size=0.2, random_state=42)

    degrees = [2, 6, 8]
    lambdas = [1e4, 1, 1e-4, 1e-8]

    ridge_rmse_train = np.zeros((len(degrees), len(lambdas)))
    ridge_rmse_test = np.zeros((len(degrees), len(lambdas)))
    lasso_rmse_train = np.zeros((len(degrees), len(lambdas)))
    lasso_rmse_test = np.zeros((len(degrees), len(lambdas)))

    for degree_idx, degree in enumerate(degrees):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Create a 1x4 grid of subplots
        for lambda_idx, lambda_val in enumerate(lambdas):
            poly_features = prep.PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train[:, np.newaxis])
            X_test_poly = poly_features.transform(X_test[:, np.newaxis])

            # Ridge Regression using scikit-learn
            ridge_model = lmdl.Ridge(alpha=lambda_val)
            ridge_model.fit(X_train_poly, y_train)
            y_train_pred_ridge = ridge_model.predict(X_train_poly)
            y_test_pred_ridge = ridge_model.predict(X_test_poly)

            # Lasso Regression using scikit-learn
            lasso_model = lmdl.Lasso(alpha=lambda_val, max_iter=10000)
            lasso_model.fit(X_train_poly, y_train)
            y_train_pred_lasso = lasso_model.predict(X_train_poly)
            y_test_pred_lasso = lasso_model.predict(X_test_poly)

            ridge_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_ridge)
            ridge_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_ridge)
            lasso_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_lasso)
            lasso_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_lasso)

            # Plot the fitted curves for both Ridge and Lasso
            X_plot = np.linspace(-10, 10, 100)
            X_plot_poly = poly_features.transform(X_plot[:, np.newaxis])

            y_plot_ridge = ridge_model.predict(X_plot_poly)
            y_plot_lasso = lasso_model.predict(X_plot_poly)

            ax = axs[lambda_idx]
            ax.scatter(X_train, y_train, color='blue', label='Train Data')
            ax.scatter(X_test, y_test, color='green', label='Test Data')
            ax.plot(X_plot, y_plot_ridge, color='red', label=f'Ridge (λ={lambda_val})')
            ax.plot(X_plot, y_plot_lasso, color='orange', linestyle='--', label=f'Lasso (λ={lambda_val})')
            ax.set_title(f'Polynomial Degree {degree} - λ={lambda_val}')
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.legend()

        plt.suptitle(f'Polynomial Degree {degree} - Regularization Comparison')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # plot RMSE
    plt.figure(figsize=(10, 6))

    for degree_idx, degree in enumerate(degrees):
        plt.plot(lambdas, ridge_rmse_test[degree_idx], marker='x', label=f'Ridge - Degree {degree}')
    plt.xscale('log')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('RMSE')
    plt.title('RMSE for Ridge at Different Polynomial Degrees')
    plt.legend()
    plt.show()

    for degree_idx, degree in enumerate(degrees):
        plt.plot(lambdas, lasso_rmse_test[degree_idx], marker='x', label=f'Lasso - Degree {degree}')

    plt.xscale('log')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('RMSE')
    plt.title('RMSE for Lasso at Different Polynomial Degrees')
    plt.legend()
    plt.show()

def test2_example():
    housing = ds.fetch_california_housing()
    X, y = housing.data, housing.target

    feature_names = housing.feature_names
    print("Feature names:", feature_names)

    X_train, X_test, y_train, y_test = mdl.train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = prep.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = lmdl.LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(metr.mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.2f}")

    coefficients = model.coef_
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.show()