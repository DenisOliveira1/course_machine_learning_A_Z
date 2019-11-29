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
# a biblioteca aplica feature scalling internamente, assim como no python
# train_set$Country = as.numeric(train_set$Country)
# test_set$Country = as.numeric(test_set$Country)
# train_set[,1:3] = scale(train_set[,1:3])
# test_set[,1:3]  = scale(test_set[,1:3])

# SVR

# Fitting SVR to the dataset
library(e1071)
reg = svm(formula = Salary ~ .,
          data = dataset,
          type = "eps-regression")
summary(reg)

# Prediction a new reuslt with SVR
x = data.frame(Level = 6.5)
Y_pred = predict(reg, newdata = x)

# Visualising the SVR results 
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(reg, newdata = dataset)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("SVR") +
  ylab("Level") +
  xlab("Salary")

# Visualising the SVR results (grid)
grid = seq(1, 10, by=0.1)
X_grid = data.frame("Level" = grid)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = X_grid$Level, y = predict(reg, newdata = X_grid)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Polynomial SVR (grid = 0.1)") +
  ylab("Level") +
  xlab("Salary")
