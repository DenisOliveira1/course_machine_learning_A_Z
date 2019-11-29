# Importing the dataset
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[3:5]

# Encoding the target feature as factor
# decision tree, random forest e naive bayes precisam dessa linha pois trabalham com factores
# dataset$Customer_Segment = factor(dataset$Customer_Segment, levels = c(1,2,3))

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
# a biblioteca NÂO aplica feature scalling internamente automaticamente, assim como no python
train_set[-3] = scale(train_set[-3])
test_set[-3]  = scale(test_set[-3])

# Aplying Kernal PCA
# install.packages("kernlab")
library(kernlab)
kpca = kpca(~.,
            data = train_set[-3],
            kernel = "rbfdot",
            features = 2)
train_set_pca = as.data.frame(predict(kpca,
                                      train_set))
test_set_pca = as.data.frame(predict(kpca,
                                    test_set))
train_set[1] = train_set_pca[1]
train_set[2] = train_set_pca[2]
test_set[1] = test_set_pca[1]
test_set[2] = test_set_pca[2]

# Logistics Regression

# Fitting Logistics Regression to the dataset
cla = glm(formula = Purchased ~ .,
          family = binomial,
          data = train_set)
cla.fit(train_set)
summary(cla)

# Predicting the Test set results
prob_pred = predict(cla, type = "response", newdata = test_set[-3])
Y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[,3], Y_pred)

# Visualising the Test set Results
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(cla, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Train set)',
     xlab = 'Kernel PC1',
     ylab = 'Kernel PC2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4','red3'))

# Visualising the Test set Results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(cla, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Kernel PC1',
     ylab = 'Kernel PC2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4','red3'))