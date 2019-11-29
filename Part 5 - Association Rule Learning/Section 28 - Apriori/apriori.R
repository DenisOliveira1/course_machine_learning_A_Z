# Importing the dataset
dataset_original = read.csv("Market_Basket_Optimisation.csv", header = F)

# Creating the sparse matrix
# sparse matrix é um dataset que contêm muitos zeros, 
# assim como no NLP para representar a frequencia das palavras
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = T)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# minimal support:
# o gráfico de frequencia seria o support
# vamos considerar somente produtos comprados ao menos 3 vezes por dia
# sabendo-se que o dataset é do periodo de 1 semana
# 3*7 = 21
# sabendo-se que o dataset tem 7501 produtos
# o support deve ser = 21/7501 = 0,002799627
# minimal confidence:
# chutar valores... default = 0.8, ou seja, 80%
# um valor de confidence muito alto vai encontrar poucas e obvias associações
# 0.8 = 0 rules
# 0.7 = 2 rules
# 0.6 = 13 rules
# 0.5 = 89 rules
# alguns produtos são associados màs regras meramente por terem um grande suporte, 
# ou seja, serem muito vendidos, como a água que esta presente em 20% das vendas
# logo, abaixando a confidence novas regras podem aparecer. O support não foi alterado para manter 
# a regra de 3 vendas diarias.

#tentando outro supprot: 4*7/7501 = 0,003732836
rules = apriori(data = dataset,
                parameter = list(support = 0.0037,
                                 confidence = 0.2))
# Visualizing the results
inspect(sort(rules, by = "lift")[1:10])