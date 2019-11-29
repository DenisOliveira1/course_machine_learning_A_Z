# Importing the dataset
dataset = read.csv("Mall_Customers.csv")
# se usar dataset[4:5]  por padrão os números já se referem as colunas
X = dataset[,4:5]

# K-Means

# Using the Elbow method to fing the optiomal number of clusters
set.seed(6)
# cria um vetor vazio
WCSS = vector()
for (i in 1:10) WCSS[i] = sum(kmeans(X,
                                     i,
                                     iter.max = 300,
                                     nstart = 10)$withinss)
plot(1:10, WCSS,
     type = "b",
     main = "Elbow method",
     xlab = "Number of clusters",
     ylab = "WCSS")

# Fitting k-means to the dataset
set.seed(29)
kmeans = kmeans(X,
                5,
                iter.max = 300,
                nstart = 10)
Y_pred = kmeans$cluster

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
         main = "K-Means",
         xlab = "Annual Income (k$)",
         ylab = "Spending Score")

