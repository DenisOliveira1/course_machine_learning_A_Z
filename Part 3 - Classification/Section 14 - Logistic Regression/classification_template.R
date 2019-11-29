# Importing the dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[3:5]

# Encoding the target feature as factor
# decision tre, random forest r naive bayes precisam dessa linha depois trabalham com factores
# dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Encoding categorical data
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
# dataset$Gender = factor(dataset$Gender,
#                         levels = c("Male","Female"),
#                         labels = c(1,2))

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente, asism como no python
train_set[,1:2] = scale(train_set[,1:2])
test_set[,1:2]  = scale(test_set[,1:2])

# Classifier

# Fitting Classifier to the dataset
summary(cla)

# Predicting the Test set results
Y_pred = predict(cla, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[,3], Y_pred)

# Visualising the Train set Results
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(cla, newdata = grid_set)
plot(set[, -3],
     main = 'Classifier (Train set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set Results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(cla, newdata = grid_set)
plot(set[, -3],
     main = 'Classifier (Test set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
