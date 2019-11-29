import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# entre [] se por o index das colunas que você quer, é um outro modo
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
import random
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(N):
    max_random = 0
    ad = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)    
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward =  dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
        total_reward = total_reward + 1
    elif reward == 0:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    
# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
