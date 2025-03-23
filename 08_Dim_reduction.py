import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection as mdl
import sklearn.preprocessing as prep
import sklearn.datasets as ds
import sklearn.metrics as metr
import sklearn.cluster as clu
import sklearn.decomposition as decom
import sklearn.linear_model as lmdl

# import seaborn as sns
# import IPython.display as dis


def load_data():
    import tensorflow.keras.datasets as tf_ds

    (X_train, y_train), (X_test, y_test) = tf_ds.fashion_mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X_flattened = X.reshape(X.shape[0], -1)

    # Display the shape of the data
    print(f"Data shape: {X_flattened.shape}")  # Should be (70000, 784)
    print(f"Labels shape: {y.shape}")
    return X, y, X_flattened

def test1():
    X, y, X_flattened = load_data()

    # Define class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Visualize a sample image
    # plt.imshow(X[0], cmap='gray')
    # plt.title(f"Label: {class_names[y[0]]}")
    # plt.axis('off')
    # plt.show()

    # Standardize the data
    scaler = prep.StandardScaler()
    X_standardized = scaler.fit_transform(X_flattened)

    # Display the shape of the standardized data
    print(f"Standardized Data Shape: {X_standardized.shape}")

    # Compute the covariance matrix
    # For large datasets, it's more efficient to compute the covariance matrix using numpy's dot product
    n_samples = X_standardized.shape[0]
    cov_matrix = np.dot(X_standardized.T, X_standardized) / (n_samples - 1)

    # Display the shape of the covariance matrix
    print(f"Covariance Matrix Shape: {cov_matrix.shape}")

    # Display the covariance matrix
    # print(f"Covariance Matrix: \n{cov_matrix}")

    # Compute eigenvalues and eigenvectors using Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(X_standardized, full_matrices=False)

    # Eigenvalues are the square of singular values divided by (n_samples - 1)
    eig_values = (S ** 2) / (n_samples - 1)

    # Eigenvectors are the transpose of V from SVD
    eig_vectors = VT.T

    # Display the shape of eigenvalues and eigenvectors
    print(f"Eigenvalues Shape: {eig_values.shape}")
    print(f"Eigenvectors Shape: {eig_vectors.shape}")

    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(eig_values[i], eig_vectors[:, i]) for i in range(len(eig_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Extract the sorted eigenvalues and eigenvectors
    sorted_eig_values = np.array([eig_pair[0] for eig_pair in eig_pairs])
    sorted_eig_vectors = np.array([eig_pair[1] for eig_pair in eig_pairs]).T

    # Display the top 10 eigenvalues
    print("Top 10 Eigenvalues:")
    for i in range(10):
        print(f"Eigenvalue {i+1}: {sorted_eig_values[i]}")

    # Select the top k eigenvectors (here we choose k=2 for visualization)
    k = 2
    matrix_w = sorted_eig_vectors[:, :k]

    # Project the standardized data onto the new feature space
    X_pca = X_standardized.dot(matrix_w)

    # Create a DataFrame with the projected data
    principal_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(k)])
    principal_df['label'] = y

    # Display the first 5 rows of the projected data
    print("Projected Data (first 5 samples):")
    print(principal_df.head())

    # Compute the explained variance ratio
    explained_variance_ratio = sorted_eig_values / np.sum(sorted_eig_values)

    # Display the explained variance ratio for the top 10 components
    print("Explained Variance Ratio for Top 10 Components:")
    for i in range(10):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f}")

    # Plot the explained variance ratio for the top 10 components
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    components = np.arange(1, 11)
    plt.bar(components, explained_variance_ratio[:10], alpha=0.7, color='blue')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio of Top 10 Principal Components')
    plt.xticks(components)
    plt.grid(axis='y')
    plt.show()

    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--', color='green')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Plot the projected data
    plt.figure(figsize=(10,8))

    for label in np.unique(y):
        indices = principal_df['label'] == label
        plt.scatter(principal_df.loc[indices, 'PC1'],
                    principal_df.loc[indices, 'PC2'],
                    s=1, alpha=0.5, label=class_names[label])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Fashion MNIST Dataset')
    plt.legend(markerscale=6)
    plt.grid()
    plt.show()

def test2():
    X, y, X_flattened = load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Initialize PCA with the desired number of components
    k = 2  # Number of principal components
    pca = decom.PCA(n_components=k)
    
    scaler = prep.StandardScaler()
    X_standardized = scaler.fit_transform(X_flattened)
    
    # Fit PCA on the standardized data
    X_pca_sklearn = pca.fit_transform(X_standardized)

    # Create a DataFrame with the projected data
    principal_df_sklearn = pd.DataFrame(X_pca_sklearn, columns=[f'PC{i+1}' for i in range(k)])
    principal_df_sklearn['label'] = y

    # Display the explained variance ratio
    print("Explained Variance Ratio by scikit-learn PCA:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    # Plot the projected data using scikit-learn PCA
    plt.figure(figsize=(10, 8))

    # No sampling - use the entire dataset
    for label in np.unique(principal_df_sklearn['label']):
        label_indices = principal_df_sklearn['label'] == label
        plt.scatter(principal_df_sklearn.loc[label_indices, 'PC1'],
                    principal_df_sklearn.loc[label_indices, 'PC2'],
                    s=1, alpha=0.5, label=class_names[label])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Fashion MNIST Dataset (scikit-learn)')
    plt.legend(markerscale=6)
    plt.grid()
    plt.show()

def test3():
    import warnings
    warnings.filterwarnings('ignore')

    # Load the Leukemia dataset from OpenML
    leukemia = ds.fetch_openml(data_id=1104, as_frame=False)
    X = leukemia.data
    y = leukemia.target

    # Convert labels to binary (0 and 1)
    le = prep.LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y_encoded))}")

    # Standardize the data
    scaler = prep.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define a logistic regression model
    model = lmdl.LogisticRegression(max_iter=1000, solver='saga')

    # Stratified k-fold cross-validation
    cv = mdl.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation without PCA
    scores_no_pca = mdl.cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    mean_score_no_pca = np.mean(scores_no_pca)

    print(f"\nCross-validation accuracy without PCA: {mean_score_no_pca:.4f}")

    # Apply PCA to reduce dimensionality
    pca = decom.PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Cross-validation with PCA
    scores_pca = mdl.cross_val_score(model, X_pca, y_encoded, cv=cv, scoring='accuracy')
    mean_score_pca = np.mean(scores_pca)

    print(f"Cross-validation accuracy with PCA (20 components): {mean_score_pca:.4f}")

    # Optional: Experiment with different numbers of components
    components = [5, 10, 20, 50]
    scores = []

    for n in components:
        pca = decom.PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        score = np.mean(mdl.cross_val_score(model, X_pca, y_encoded, cv=cv, scoring='accuracy'))
        scores.append(score)
        print(f"Accuracy with PCA ({n} components): {score:.4f}")

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(components, scores, marker='o', label='With PCA')
    plt.axhline(y=mean_score_no_pca, color='r', linestyle='--', label='Without PCA')
    plt.title('Model Accuracy vs. Number of PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cross-validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def test4():
    mnist, labels = ds.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    mnist = mnist.astype('float32') / 255.0
    labels = labels.astype(int)
    num_pcs = 20

    def pca(X, num_components):
        X_meaned = X - np.mean(X, axis=0)
        covariance_matrix = np.cov(X_meaned, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, :num_components]
        X_reduced = np.dot(X_meaned, eigenvector_subset)
        return X_reduced, eigenvector_subset

    mnist_reduced, eigenvector_subset = pca(mnist, num_pcs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mnist_reduced[:, 0], mnist_reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


    mnist_decompressed = np.dot(mnist_reduced, eigenvector_subset.T) + np.mean(mnist, axis=0)

    def visualize_decompression(original, decompressed, img_shape, num_images=5, title=""):
        original = original.reshape(-1, *img_shape)
        decompressed = decompressed.reshape(-1, *img_shape)

        plt.figure(figsize=(10, 4))

        for i in range(num_images):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(original[i])
            plt.axis('off')

            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(decompressed[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()


    visualize_decompression(mnist[:5], mnist_decompressed[:5], img_shape=(28, 28), title="MNIST")