# Importing the dataset
dataset_original = read.csv("Market_Basket_Optimisation.csv", header = F)

# Creating the sparse matrix
# sparse matrix é um dataset que contêm muitos zeros, 
# assim como no NLP para representar a frequencia das palavras
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = T)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
sets = eclat(data = dataset,
                parameter = list(support = 0.003,
                                 minlen = 2))
# Visualizing the results
inspect(sort(sets, by = "support")[1:10])