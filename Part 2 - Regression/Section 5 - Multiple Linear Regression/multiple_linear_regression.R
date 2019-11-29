# Importing the dataset
dataset = read.csv("50_Startups.csv")

# Encoding categorical data
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
# tem que manter como factor para transformar em dummy variable e trata o dummy variale trap, ambos automaticamente 
dataset$State = factor(dataset$State,
                         levels = c("New York","California","Florida"),
                         labels = c(1,2,3))

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# a biblioteca aplica feature scalling internamente, assim como no python
# train_set$Country = as.numeric(train_set$Country)
# test_set$Country = as.numeric(test_set$Country)
# train_set[,1:3] = scale(train_set[,1:3])
# test_set[,1:3]  = scale(test_set[,1:3])

# Multiple Linear Regression

# Fitting Multiple Linear Regression to the Training set
# no fit o algoritmo estuda e aprende a corelação entre as variáveis independentes e a dependente
# a formula recebe como paremetros a correlção entre as colunas do data
#reg = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
#         data = train_set)
reg = lm(formula = Profit ~ .,
         data = train_set)
# summary(reg)

# Predicting the Test set results
# tranformei em dataframe somente para visualizar no Environment
Y_pred = as.data.frame(predict(reg, newdata = test_set))

# Predicting the Test set results with the best predictor
# ambas não deram o mesmo resultado como dito no vídeo
reg = lm(formula = Profit ~ R.D.Spend ,
         data = train_set)
# summary(reg)
Y_pred_best_predictor = as.data.frame(predict(reg, newdata = test_set))

# Building the optimal model using Backward Elimination
# para fazer Backward Elimination se usa a mesma função, porem com um diferente sistema de parâmetros
reg = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
         data = dataset)
summary(reg)
# remove colunas enquanto o maior p for maior que o limite selecionado, nesse exemplo 0.05, ou seja, 5%
# remove State
reg = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
         data = dataset)
summary(reg)
# remove Administration
reg = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
         data = dataset)
summary(reg)
# remove Marketing.Spend
reg = lm(formula = Profit ~ R.D.Spend,
         data = dataset)
summary(reg)

# Predicting the optimal Test set results
reg = lm(formula = Profit ~ R.D.Spend ,
         data = train_set)
Y_opt_pred = as.data.frame(predict(reg, newdata = test_set))
