# Importing the dataset
dataset = read.csv("Wine.csv")
#dataset = dataset[3:5]

# Encoding the target feature as factor
# decision tree, random forest e naive bayes precisam dessa linha pois trabalham com factores
dataset$Customer_Segment = factor(dataset$Customer_Segment, levels = c(1,2,3))

# Encoding categorical data
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
# dataset$Gender = factor(dataset$Gender,
#                         levels = c("Male","Female"),
#                         labels = c(1,2))

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente, assim como no python
train_set[-14] = scale(train_set[-14])
test_set[-14]  = scale(test_set[-14])

# Aplying PCA
library(caret)
library(e1071)
pca = preProcess(x = train_set[-14],
               method = "pca",
               #thresh = 0.6, pega quantos componentes forem necessarios para representar 60% da variancia
               pcaComp = 2)
train_set = predict(pca,
                  train_set)
train_set = train_set[c(2,3,1)]
test_set = predict(pca,
                 test_set)
test_set = test_set[c(2,3,1)]

# Logistics Regression

# Fitting Logistics Regression to the dataset
library(nnet)
glm.fit = multinom(Customer_Segment ~ .,
             data = train_set)
summary(glm.fit)

# Predicting the Test set results
Y_pred  = predict(glm.fit, type = "class", newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[,3], Y_pred)

# Visualising the Test set Results
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(glm.fit, type = 'class', newdata = grid_set)
plot(set[, -3],
   main = 'Logistic Regression (Train set)',
   xlab = 'PC1',
   ylab = 'PC2',
   xlim = range(X1),
   ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', ifelse(y_grid == 2, "deepskyblue", 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', ifelse(set[, 3] == 2,'blue3','red3')))

# Visualising the Test set Results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(glm.fit, type = 'class', newdata = grid_set)
plot(set[, -3],
   main = 'Logistic Regression (Test set)',
   xlab = 'PC1',
   ylab = 'PC2',
   xlim = range(X1),
   ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', ifelse(y_grid == 2, "deepskyblue", 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', ifelse(set[, 3] == 2,'blue3','red3')))