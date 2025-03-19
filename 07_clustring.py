import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection as mdl
import sklearn.preprocessing as prep
import sklearn.linear_model as lmdl
import sklearn.datasets as ds
import sklearn.metrics as metr
import sklearn.cluster as clu
import sklearn.decomposition as decom
import IPython.display as dis

def test1():
    home_data = pd.read_csv('./asset/housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
    print(home_data.head())

    sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')
    plt.show()

    X_train, X_test, y_train, y_test = mdl.train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

    X_train_norm = prep.normalize(X_train)
    X_test_norm = prep.normalize(X_test)

    kmeans = clu.KMeans(n_clusters = 3, random_state = 0, n_init='auto')
    kmeans.fit(X_train_norm)

    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
    plt.show()

    sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'], hue=kmeans.labels_)
    plt.show()

    K = range(2, 8)
    fits = []

    for k in K:
        # train the model for current value of k on training data
        model = clu.KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)

        # append the model to fits
        fits.append(model)

    f, axes = plt.subplots(2, 3, figsize=(18, 10))
    f.suptitle('K-Means Clustering', fontsize=24)
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[0].labels_, ax=axes[0,0])
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[1].labels_, ax=axes[0,1])
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[2].labels_, ax=axes[0,2])
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[3].labels_, ax=axes[1,0])
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[4].labels_, ax=axes[1,1])
    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[5].labels_, ax=axes[1,2])
    plt.show()

    kmeans = clu.KMeans(n_clusters = 5, random_state = 0, n_init='auto')
    kmeans.fit(X_train_norm)

    sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
    plt.show()

    sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'], hue=kmeans.labels_)
    plt.show()

def test2():
    players = pd.read_csv("./asset/players_22.csv")
    # print(players.head())
    features = ["overall", "potential", "wage_eur", "value_eur", "age"]
    players = players.dropna(subset=features)
    data = players[features].copy()
    print(data.head())

    data = ((data - data.min()) / (data.max() - data.min()))
    print(data.describe())

    def random_centroids(data, k):
        centroids = []
        for i in range(k):
            centroid = data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    centroids = random_centroids(data, 5)

    def get_labels(data, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
        return distances.idxmin(axis=1)

    labels = get_labels(data, centroids)
    labels.value_counts()
    def new_centroids(data, labels, k):
        centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
        return centroids
    def plot_clusters(data, labels, centroids, iteration):
        pca = decom.PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids.T)
        dis.clear_output(wait=True)
        plt.title(f'Iteration {iteration}')
        plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
        plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
        plt.show()

    max_iterations = 100
    centroid_count = 3

    centroids = random_centroids(data, centroid_count)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, centroid_count)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    print(centroids)
    kmeans = clu.KMeans(3)
    kmeans.fit(data)

def test3():
    iris = ds.load_iris()
    X = iris.data
    y = iris.target

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Inertia (Elbow Method)
    def compute_inertia(X, max_k=10):
        inertia_vals = []
        for k in range(1, max_k + 1):
            kmeans = clu.KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia_vals.append(kmeans.inertia_)
        return inertia_vals

    inertia_vals = compute_inertia(X, max_k=10)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia_vals, marker='o', color="red")
    plt.axvline(x=3, ls='--')
    plt.title('Inertia (Elbow Method)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()


    sil = []
    max_k = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, max_k+1):
        kmeans = clu.KMeans(n_clusters = k)
        kmeans.fit(X)
        labels = kmeans.labels_
        sil.append(metr.silhouette_score(X, labels, metric = 'euclidean'))

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), sil, marker='o')
    plt.title('Silhouette Score For Different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Davies-Bouldin Index
    def compute_db_index(X, max_k=10):
        db_vals = []
        for k in range(2, max_k + 1):
            kmeans = clu.KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            db_vals.append(metr.davies_bouldin_score(X, labels))
        return db_vals

    db_vals = compute_db_index(X, max_k=10)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), db_vals, marker='o', color='red')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('DB Index')
    plt.show()

def test4():
    X, y = ds.make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42)

    # Plot the true clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=50)
    plt.title("True Clusters (Generated by make_blobs)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    random_state = 42
    n_clusters = 3
    max_iter = 10
    np.random.seed(random_state)

    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    centroids_history = [centroids.copy()]

    for i in range(max_iter):
        labels = metr.pairwise_distances_argmin(X, centroids)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(n_clusters)])
        centroids_history.append(new_centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=50)
    plt.scatter(centroids_history[:,0], centroids_history[:,1])
    plt.title("True Clusters (Generated by make_blobs)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def test5():
    # Generate a dataset with 3 clusters
    X, y = ds.make_blobs(n_samples=450, centers=3, random_state=42, cluster_std=2.0)

    # Plot the dataset without clustering information
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30)
    plt.title("Initial Data")
    plt.show()

    n_clusters = 3
    max_iter=300
    # Randomly initialize centroids from the data points
    rng = np.random.RandomState(40)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centroids = X[i]

    for i in range(max_iter):
        # Step 1: Assign points to the nearest centroid
        labels = metr.pairwise_distances_argmin(X, centroids)

        # Step 2: Plot the points with their current cluster assignments
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
        plt.title(f"Iteration {i+1}")
        
        # Save the plot as an image
        plt.savefig(f"kmeans_iter_{i+1}.png")
        plt.show()

        # Step 3: Update centroids
        new_centroids = np.array([X[labels == j].mean(0) for j in range(n_clusters)])

        # Break if the centroids have stopped moving
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

