# Importing the dataset
dataset = read.csv("Salary_Data.csv")

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca aplica feature scalling internamente, assim como no python
# train_set$Country = as.numeric(train_set$Country)
# test_set$Country = as.numeric(test_set$Country)
# train_set[,1:3] = scale(train_set[,1:3])
# test_set[,1:3]  = scale(test_set[,1:3])
# Simple Linear Regression

# Fitting Simple Linear Regression to the Training set
# no fit o algoritmo estuda e aprende a corelação entre as variáveis independentes e a dependente
# a formula recebe como paremetros a correlção entre as colunas do data
reg = lm(formula = Salary ~ YearsExperience,
         data = train_set)
# summary(reg)

# Predicting the Test set results
# tranformei em dataframe somente para visualizarno Environment
Y_pred = as.data.frame(predict(reg, newdata = test_set))

# Visualising the Training set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary), 
             colour = "red") +
  geom_line(aes(x = train_set$YearsExperience,  predict(reg, newdata = train_set)),
            colour = "blue") +
  ggtitle("Salary x Experience (train set)") +
  xlab("Years of Experience") +
  ylab("Salary")

# Visualising the Test set results
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = "green") +
  geom_line(aes(x = train_set$YearsExperience,  predict(reg, newdata = train_set)),
            colour = "blue") +
  ggtitle("Salary x Experience (test set)") +
  xlab("Years of Experience") +
  ylab("Salary")

# Visualising the Train and Test set results
ggplot() + 
  geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary), 
             colour = "red") +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = "green") +
  geom_line(aes(x = train_set$YearsExperience,  predict(reg, newdata = train_set)),
            colour = "blue") +
  ggtitle("Salary x Experience (train and test set)") +
  xlab("Years of Experience") +
  ylab("Salary")
