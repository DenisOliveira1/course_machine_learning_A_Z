import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# Creating the sparse matrix
# sparse matrix é um dataset que contêm muitos zeros, 
# assim como no NLP para representar a frequencia das palavras
transactions = []
for i in range(len(dataset)):# para cada linha 
    transactions.append([str(dataset.values[i,j]) for j in range(dataset.shape[1]) if str(dataset.values[i,j]) != "nan"])
# número de colunas pode ser obtido por len(dataset.values[0,:]) ou dataset.shape[1]

# Training Apriori on the dataset
# minimal support:
# vamos considerar somente produtos comprados ao menos 3 vezes por dia
# sabendo-se que o dataset é do periodo de 1 semana
# 3*7 = 21
# sabendo-se que o dataset tem 7501 produtos
# o support deve ser = 21/7501 = 0,002799627
# minimal confidence:
# chutar valores... 0.8, significa que 80% das vezes que o conjunto
# de items A é comprando B também é
rules = apriori(transactions,
                min_support = 0.002799627,
                min_confidence = 0.2,
                min_lift = 3,
                min_length = 2)

# Visualizing the results
results = list(rules)

for i in range(20):
    print("["+str(i)+"]")
    print("rule:",results[i][0])
    print("support:",results[i][1])
    print("confidence:",results[i][2][0][2])
    print("lift:",results[i][2][0][3])