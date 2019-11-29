# Importing the dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[2:5]

# Encoding categorical data
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
dataset$Gender = factor(dataset$Gender,
                        levels = c("Male","Female"),
                        labels = c(1,2))

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca NÂO aplica feature scalling internamente automaticamente, asism como no python
train_set$Gender = as.numeric(train_set$Gender)
test_set$Gender = as.numeric(test_set$Gender)
train_set[,1:3] = scale(train_set[,1:3])
test_set[,1:3]  = scale(test_set[,1:3])

# Classifier

# Fitting Classifier to the dataset
summary(cla)

# Predicting the Test set results
Y_pred = predict(cla, newdata = test_set[-4])

# Making the Confusion Matrix
cm = table(test_set[,4], Y_pred)

# Visualising the Train set Results
library(ElemStatLearn)
set = train_set[,-1]
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
X0 = rep(c(0), times = c(length(X1)))
# o fato de gerar 0 fazer o predict de todo o grid causa incosistencia no grafico, não se deve fazer isso, é como forçar um valor
grid_set = expand.grid(X1,X2)
grid_set$Gender = X0
colnames(grid_set) = c('Age', 'EstimatedSalary',"Gender")
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
set = test_set[,-1]
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
# o fato de gerar 0 fazer o predict de todo o grid causa incosistencia no grafico, não se deve fazer isso, é como forçar um valor
X0 = rep(c(0), times = c(length(X1)))
grid_set = expand.grid(X1,X2)
grid_set$Gender = X0
colnames(grid_set) = c('Age', 'EstimatedSalary',"Gender")
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


# no knn

# Visualising the Train set Results
library(ElemStatLearn)
set = train_set[,-1]
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
X0 = rep(c(0), times = c(length(X1)))
grid_set = expand.grid(X1,X2)
grid_set$Gender = X0
grid_set = grid_set[,c(3,1,2)]
colnames(grid_set) = c("Gender",'Age', 'EstimatedSalary')
y_grid = knn(train = train_set[,-4],
             test = grid_set,
             cl = train_set[, 4],
             k = 5)
plot(set[, -3],
     main = 'KNN (Train set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
# logo é preciso eliminar a primeira coluna aqui (Gender), já que essa função odentifica a coluna 1 como x e a coluna 2 como y
points(grid_set[,-1], pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set Results
library(ElemStatLearn)
set = test_set[,-1]
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
X0 = rep(c(0), times = c(length(X1)))
grid_set = expand.grid(X1,X2)
grid_set$Gender = X0
# no knn as colunas do teste devem estar iguais ao do treino
grid_set = grid_set[,c(3,1,2)]
colnames(grid_set) = c("Gender",'Age', 'EstimatedSalary')
y_grid = knn(train = train_set[,-4],
             test = grid_set,
             cl = train_set[, 4],
             k = 5)
plot(set[, -3],
     main = 'KNN (Test set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
# logo é preciso eliminar a primeira coluna aqui (Gender), já que essa função odentifica a coluna 1 como x e a coluna 2 como y
points(grid_set[,-1], pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))