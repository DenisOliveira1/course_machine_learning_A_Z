import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Hierachical Clustering

# Using the dendrogram to find the optiomal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

# Fitting the Hierachical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,
                             affinity = "euclidean",
                             linkage = "ward")

Y_pred = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[Y_pred == 0, 0],X[Y_pred == 0, 1], s = 100, c = "red", label = "Careful")
plt.scatter(X[Y_pred == 1, 0],X[Y_pred == 1, 1], s = 100, c = "blue", label = "Standard")
plt.scatter(X[Y_pred == 2, 0],X[Y_pred == 2, 1], s = 100, c = "green", label = "Target")
plt.scatter(X[Y_pred == 3, 0],X[Y_pred == 3, 1], s = 100, c = "cyan", label = "Careless")
plt.scatter(X[Y_pred == 4, 0],X[Y_pred == 4, 1], s = 100, c = "magenta", label = "Sensible")
plt.title("Hierachical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()