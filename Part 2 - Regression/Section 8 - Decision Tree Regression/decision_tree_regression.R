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
# decision tree é idependente de feature scalling, ou seja, seu resultado com ou sem featuring scale é o mesmo
# train_set$Country = as.numeric(train_set$Country)
# test_set$Country = as.numeric(test_set$Country)
# train_set[,1:3] = scale(train_set[,1:3])
# test_set[,1:3]  = scale(test_set[,1:3])

# Decision Tree Regression

# Fitting Decision Tree Regression to the dataset
library(rpart)
reg = rpart(formula = Salary ~ .,
          data = dataset,
          method = "poisson",
          control = rpart.control(minsplit = 1, 
                                  cp = 0))
summary(reg)

# Prediction a new reuslt with Decision Tree Regression
# não consegui chegar no mesmo valor que o python...diferença grande
x = data.frame(Level = 6.5)
Y_pred = as.numeric(predict(reg, newdata = x))

# Visualising the Decision Tree Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(reg, newdata = dataset)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Decision Tree Regression") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Decision Tree Regression results (grid)
grid = seq(1, 10, by=0.01)
X_grid = data.frame("Level" = grid)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = X_grid$Level, y = predict(reg, newdata = X_grid)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(reg, newdata = data.frame(Level = 6.5))),
             colour = "green") +
  ggtitle("Decision Tree Regression (grid = 0.01)") +
  ylab("Level") +
  xlab("Salary")
