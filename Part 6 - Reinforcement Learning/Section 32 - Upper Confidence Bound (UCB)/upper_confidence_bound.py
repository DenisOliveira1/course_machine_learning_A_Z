import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# entre [] se por o index das colunas que você quer, é um outro modo
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
import math
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# antes de entrar no loop, as 10 primeiras seleções serão 1 de cada
# ad (coluna do dataset), para coletar dados iniciais para só então ter dados
# suficientes para realizar as seleções
for n in range(N):
    max_upper_bound = 0
    ad = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(1.5 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # nas oprimeira interações força cada uma das ads a ter o maximo
            # upper_bound e ser eleita 
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward =  dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
