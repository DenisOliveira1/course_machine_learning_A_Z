# Importing dataset
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
N = nrow(dataset)
d = ncol(dataset)
ads_selected = vector()
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N){
  max_random = 0
  ad = 0
  for (i in 1:d){
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_rewards_1[i] + 1,
                        shape2 = numbers_of_rewards_0[i] + 1)
    if (random_beta  > max_random){
      max_random = random_beta
      ad = i
    }
  }
  ads_selected = c(ads_selected, ad)
  reward = dataset[n,ad]
  if (reward == 1){
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    total_reward = total_reward + reward
  }
  else if (reward == 0){
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
}

# Visualizing the results
hist(ads_selected,
     col = "blue",
     main = "Histogram of ads selections",
     xlab = "Ads",
     ylab = "Number of times each ad was selected")