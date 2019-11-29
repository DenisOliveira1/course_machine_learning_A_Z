# Importing the dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5]

# Hierachical Clustering

# Using the dendrogram to find the optiomal number of clusters
dendrogram = hclust(dist(X,method = "euclidean"),
                    method = "ward.D")
plot(dendrogram,
     main = "Demdrogram",
     ylab = "Customers",
     xlab = "Euclidean distance")

# Fitting the Hierachical Clustering to the dataset

hc = hclust(dist(X, method = "euclidean"),
       method = "ward.D")
Y_pred = cutree(hc, 5)

# Visualizing the clusters
library(cluster)
clusplot(X,
         Y_pred,
         lines = 0,
         shade = F,
         color = T,
         labels = 2,
         plotchar = F,
         span = T,
         main = "Hierachical Clustering",
         xlab = "Annual Income (k$)",
         ylab = "Spending Score")
