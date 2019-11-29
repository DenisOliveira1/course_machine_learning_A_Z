# Importing the dataset
dataset = read.csv("Data.csv")

# Taking care of missing data
# mean é uma função do R onde na.rm é um parâmetro que remove todos os NA (NaN) antes de calcular a média
# a funcao ifelse recebe condição, return se true, return se false
# se a coluna tiver um NA vai dar true e vai retornar a média dessa coluna ignorando os NA
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encoding categorical data
# c é um vetor no R
# com esse processo a coluna assume valores do tipo factor, e não númericos, apesar de serem números
dataset$Country = factor(dataset$Country,
                        levels = c("France","Spain","Germany"),
                        labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                         levels = c("No","Yes"),
                         labels = c(1,2))

# Splitting the dataset in train set and test set
# SplitRatio se refere ao train
# library é um modo de importar um package via código, não necessitando clicar nele no rstudio
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# ao contrário do vídeo eu tranformei em numeric Country fiz o scale para ficar igual o que foi feito em python
train_set$Country = as.numeric(train_set$Country)
test_set$Country = as.numeric(test_set$Country)
train_set[,1:3] = scale(train_set[,1:3])
test_set[,1:3]  = scale(test_set[,1:3])
