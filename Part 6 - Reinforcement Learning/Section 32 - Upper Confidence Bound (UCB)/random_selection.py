import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# entre [] se por o index das colunas que você quer, é um outro modo
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Random Selection
import random
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
total_reward = 0
for n in range(N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward = total_reward + reward

# Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()