import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# K-Means

# Using the Elbow method to fing the optiomal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,
                    init = "k-means++",
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# Fitting k-means to the dataset
kmeans = KMeans(n_clusters = 5,
                    init = "k-means++",
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
Y_pred = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[Y_pred == 0, 0],X[Y_pred == 0, 1], s = 100, c = "red", label = "Careful")
plt.scatter(X[Y_pred == 1, 0],X[Y_pred == 1, 1], s = 100, c = "blue", label = "Standard")
plt.scatter(X[Y_pred == 2, 0],X[Y_pred == 2, 1], s = 100, c = "green", label = "Target")
plt.scatter(X[Y_pred == 3, 0],X[Y_pred == 3, 1], s = 100, c = "cyan", label = "Careless")
plt.scatter(X[Y_pred == 4, 0],X[Y_pred == 4, 1], s = 100, c = "magenta", label = "Sensible")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Centroids")
plt.title("K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
