# Importing the dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Splitting the dataset in train set and test set
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Profit, SplitRatio = 0.8)
# train_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature scaling
# decision tree é idependente de feature scalling, ou seja, seu resultado com ou sem featuroing scale é o mesmo
# train_set$Country = as.numeric(train_set$Country)
# test_set$Country = as.numeric(test_set$Country)
# train_set[,1:3] = scale(train_set[,1:3])
# test_set[,1:3]  = scale(test_set[,1:3])

# Random Forest Regression

# Fitting Random Forest Regression to the dataset
# foi o algorimito com melhor previsão, com 500 árvores
library(randomForest)
set.seed(1234)
reg = randomForest(x = dataset[1],# dataframe pois é uma parte de um dataframe
                   y = dataset$Salary,# vetor
                   ntree = 500)
summary(reg)

# Prediction a new reuslt with Random Forest Regression
x = data.frame(Level = 6.5)
Y_pred = as.numeric(predict(reg, newdata = x))

# Visualising the Random Forest Regression results 
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(reg, newdata = dataset)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Random Forest Regression") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Random Forest Regression results (grid)
grid = seq(1, 10, by=0.01)
X_grid = data.frame("Level" = grid)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = X_grid$Level, y = predict(reg, newdata = X_grid)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Random Forest Regression (grid = 0.01)") +
  ylab("Level") +
  xlab("Salary")
