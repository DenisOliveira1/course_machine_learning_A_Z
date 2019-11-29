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

# XGBoost

# Fitting XGBoost to the dataset
library(xgboost)
cla = xgboost(as.matrix(train_set[-11]),
              train_set[11]$Exited,
              nrounds = 10)

# Predicting the Test set results
Y_prob = predict(cla, as.matrix(test_set[-11]))
Y_pred = (Y_prob > 0.5)

# Making the Confusion Matrix
cm = table(test_set[,11], Y_pred)
accuracy = (1540+195)/2000

# Applying k-Fold Cross Validation
# variáveis dentro de funções não aparecem no global environment
library(caret)
folds = createFolds(train_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  train_fold = train_set[-x, ]
  test_fold = train_set[x, ]
  classifier = xgboost(data = as.matrix(train_set[-11]),
                       label = train_set$Exited,
                       nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
mean(as.numeric(cv))