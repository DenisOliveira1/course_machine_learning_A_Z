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

# Regression

# Fitting Regression to the dataset
summary(reg)

# Prediction a new reuslt with Regression
Y_pred = predict(reg, newdata = data.frame(Level = 6.5))

# Visualising the Regression results 
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(reg, newdata = dataset)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Regression") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Polynomial Regression results (grid)
grid = seq(1, 10, by=0.1)
X_grid = data.frame("Level" = grid)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = X_grid$Level, y = predict(reg, newdata = X_grid)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Polynomial Regression (grid = 0.1)") +
  ylab("Level") +
  xlab("Salary")
