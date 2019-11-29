# Importing the dataset
dataset = read.csv("Churn_Modelling.csv")
dataset = dataset[4:14]

# Encoding the target feature as factor
# decision tre, random forest r naive bayes precisam dessa linha depois trabalham com factores
# dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Encoding categorical data
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
dataset$Geography = factor(dataset$Geography,
                       levels = c("France","Spain","Germany"),
                       labels = c(1,2,3))
dataset$Gender = factor(dataset$Gender,
                        levels = c("Male","Female"),
                        labels = c(1,2))

# essa biblioteca requere valores númericos e não factores
dataset$Geography = as.numeric(dataset$Geography)
dataset$Gender = as.numeric(dataset$Gender)

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente, assim como no python
train_set[,-11] = scale(train_set[,-11])
test_set[,-11]  = scale(test_set[,-11])

# ANN

# Fitting the ANN to the dataset
# essa biblioteca instalada pelo env nao funcionou
# com install.packages('h2o') deu certo
library(h2o)
h2o.init(nthreads = -1)
# dataframe para h2o
# hidden contem o número de nós em cada layer
# Dica: número de nós nas hidden layers é calculado como a media de:
# input nodes = 10
# output nodes = 1 (saida binaria)
# 10+1/2 = 5.5 = 6
cla = h2o.deeplearning(y = "Exited",
                       training_frame = as.h2o(train_set), 
                       activation = "Rectifier",
                       hidden = c(6,6),
                       epochs = 100,
                       train_samples_per_iteration = -2)

# Predicting the Test set results
Y_prob = as.vector(h2o.predict(cla, newdata = as.h2o(test_set[-11])))
Y_pred = (Y_prob > 0.5)

# Making the Confusion Matrix
cm = table(test_set[,11], Y_pred)
accuracy = (1533+193)/2000

# Desconectar do servidor
h2o.shutdown()