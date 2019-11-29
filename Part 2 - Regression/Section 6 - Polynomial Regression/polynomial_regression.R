# Importing the dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[2:3]

# Splitting the dataset in train set and test set
# o dataset é muito pequeno e precisamos de toda informação para fazer as melhores previsões possiveis
# por isso não haverá split
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

# Polynomial Linear Regression

# Fitting Linear Regression to the dataset just for comparing
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
summary(lin_reg)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
poly_reg = lm(formula = Salary ~ .,
             data = dataset)
summary(poly_reg)
      
# Visualising the Linear Regression results 
# é passado o dataset como newdata no predict linear. Mesmo o dataset tendo sido alterado (adicionado colunas),
# essas colunas vão ser ignoradas, pois não estavam presentes no momento da criação do lin_reg.
# isso pode ser visto no: summary(lin_reg)
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
            colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(lin_reg, newdata = data.frame(Level = 6.5))),
            colour = "green") +
  ggtitle("Simple Linear Regression") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Polynomial Regression results (degree = 2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = "blue") +
  ggtitle("Polynomial Regression (degree = 2)") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Polynomial Regression results (degree = 4)
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = "blue") +
  
  ggtitle("Polynomial Regression (degree = 4)") +
  ylab("Level") +
  xlab("Salary")

# Visualising the Polynomial Regression results (degree = 4 + grid)
grid = seq(1, 10, by=0.1)
X_grid = data.frame("Level" = grid,
                    "Level2" = grid^2,
                    "Level3" = grid^3,
                    "Level4" = grid^4)

poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +
  geom_line(aes(x = X_grid$Level, y = predict(poly_reg, newdata = X_grid)),
            colour = "blue") +
  geom_point(aes(x = 6.5, y = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                                     Level2 = 6.5^2,
                                                                     Level3 = 6.5^3,
                                                                     Level4 = 6.5^4))),
             colour = "green") +
  ggtitle("Polynomial Regression (degree = 4 + grid)") +
  ylab("Level") +
  xlab("Salary")

# Prediction a new reuslt with Linear Regression
pred_lin = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Prediction a new reuslt with Polyminal Regression (degree = 4 + grid)
pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                  Level2 = 6.5^2,
                                                  Level3 = 6.5^3,
                                                  Level4 = 6.5^4))
