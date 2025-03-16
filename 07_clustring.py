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


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, normalize
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans

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

test1()
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

